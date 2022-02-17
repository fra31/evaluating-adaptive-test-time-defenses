import torch
import argparse
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union
from functools import partial
import time

#from autoattack import AutoAttack
import autoattack
import other_utils
from torch import nn
#from tqdm import tqdm
import sys
sys.path.append('../')


from robustbench.utils import clean_accuracy
from apgd_tta import apgd_train as apgd_tta
from utils_tta import load_dataset, get_logits, get_wc_acc, eval_fast

from models import load_model
import criterions
from defenses import purify





aux_dict = {'pi': criterions.pi_criterion}


class PurifiedModel():
    def __init__(self, model, purifier, return_interm=False):
        assert not model.training
        self.model = model
        self.purifier = purifier
        self.return_interm = return_interm
        self.training = self.model.training
    
    def __call__(self, x):
        with torch.enable_grad():
            x_pfy, interm_x = self.purifier(self.model, x.clone())
        delta = x_pfy - x.clone()
        if not self.return_interm:
            return self.model(x + delta)
        else:
            return self.model(x + delta), interm_x.permute([1, 0, 2, 3, 4])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='Carmon2019Unlabeled')
    parser.add_argument('--threat_model',
                        type=str,
                        default='Linf',
                        #choices=[x.value for x in ThreatModel]
                        )
    parser.add_argument('--dataset',
                        type=str,
                        default='imagenet',
                        #choices=[x.value for x in BenchmarkDataset]
                        )
    parser.add_argument('--eps', type=float, default=8 / 255)
    parser.add_argument('--n_ex',
                        type=int,
                        default=100,
                        help='number of examples to evaluate on')
    parser.add_argument('--batch_size',
                        type=int,
                        default=500,
                        help='batch size for evaluation')
    parser.add_argument('--data_dir',
                        type=str,
                        default='../rb-imagenet/robustbench',
                        help='where to store downloaded datasets')
    parser.add_argument('--model_dir',
                        type=str,
                        default='../rb-imagenet/robustbench/models',
                        help='where to store downloaded models')
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help='device to use for computations')
    parser.add_argument('--to_disk', type=bool, default=True)
    parser.add_argument('--auxiliary', type=str, default='pi')
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_imgs', action='store_true')
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--eot_iter', type=int, default=1)
    parser.add_argument('--pfy_delta', type=float, default=0)
    parser.add_argument('--pfy_iter', type=int, default=5)
    parser.add_argument('--pfy_step_size', type=float, default=4.)
    parser.add_argument('--eot_test', type=int, default=1)
    parser.add_argument('--attack', type=str) #default='eval_fast'
    parser.add_argument('--n_iter', type=int, default=10)
    args = parser.parse_args()

    return args


#


def main(args: Namespace) -> None:
    model = load_model(args.model_name,
                       save_dir=args.model_dir,
                       dataset=args.dataset)

    model.eval()
    device = torch.device(args.device)
    model.to(device)
    
    aux_crit = aux_dict[args.auxiliary]
    
    #
    
    if args.eps > 1. and args.threat_model == 'Linf':
        args.eps /= 255.
        args.pfy_delta /= 255.
        args.pfy_step_size /= 255.
    #device = torch.device(args.device)
    savedir = f'./results/{args.dataset}/{args.threat_model}/{args.model_name}'
    other_utils.makedir(savedir)

    x_test, y_test = load_dataset(args.dataset, args.n_ex, device=device,
        data_dir=args.data_dir)
    
    pfy = partial(purify, defense_mode='pgd_linf', delta=args.pfy_delta,
        step_size=args.pfy_step_size, num_iter=args.pfy_iter, randomize=args.dynamic,
        aux_criterion=aux_crit, return_interm=True)

    purified_model = PurifiedModel(model, pfy)
    
    use_basemodel = args.pfy_delta == 0
    if use_basemodel:
        print('using base model')
    
    model_test = [model, purified_model][int(~use_basemodel)]
    
    for _ in range(4 * int(args.dynamic) + 1):
        for _ in range(1): #args.eot_test
            acc = clean_accuracy(model_test, x_test, y_test)
            print(f'clean accuracy={acc:.1%}')
    if args.data_path is None and args.attack is None:
        sys.exit()
    
    if args.data_path is None:
        if args.attack == 'eval_fast':
            x_adv = eval_fast(model_test, x_test, y_test, args.threat_model, args.eps,
                savedir, bs=args.batch_size, eot_iter=args.eot_iter)
        elif args.attack == 'apgd_tta':
            purified_model.return_interm = True
            _, _, _, x_adv = apgd_tta(model_test, x_test, y_test,
                args.threat_model, args.eps,
                n_iter=args.n_iter, use_interm=True, is_train=False,
                verbose=True, loss=args.loss)
            purified_model.return_interm = False
        elif args.attack == 'square':
            x_adv = eval_with_square(model_test, x_test, y_test, eps=args.eps,
                savedir=savedir, seed=None, n_iter=args.n_iter,
                loss=args.loss, opt_loss=True)
        elif args.attack == 'fgsm':
            _, _, _, x_adv = apgd_tta(model, x_test, y_test,
                args.threat_model, args.eps,
                n_iter=1, use_interm=False, is_train=False,
                verbose=True, loss=args.loss)
        elif args.attack == 'clean':
            x_adv = x_test.clone()
    
    else:
        print(f'test_points {args.data_path}')
        x_adv = torch.load(args.data_path)
        other_utils.check_imgs(x_test, x_adv, norm=args.threat_model)
    
    # eval with base model
    if not isinstance(x_adv, list):
        startt = time.time()
        for _ in range(args.eot_test):
            acc = clean_accuracy(model, x_adv, y_test, batch_size=args.batch_size)
            print(f'rob accuracy (base model)={acc:.1%}')
        evalt = time.time() - startt
        print(f'[base model] nruns={args.eot_test} avg. time={evalt / args.eot_test:.3f}')
    
    # eval with base model and purification
    for _ in range(4 * int(args.dynamic) * 0 + 1):
        #x_pfy = purify(x_adv
        if not isinstance(x_adv, list):
            startt = time.time()
            for _ in range(args.eot_test):
                acc = clean_accuracy(purified_model, x_adv, y_test, batch_size=args.batch_size)
                print(f'purified accuracy={acc:.1%}')
            evalt = time.time() - startt
            print(f'[model with purification] def_iters={args.pfy_iter}' + \
                f' nruns={args.eot_test} avg. time={evalt / args.eot_test:.3f}')
    
    
    
    
    if args.save_imgs:
        if not args.data_path in ['multiple']:
            dets = '2000x1'
            torch.save(x_adv, f'{savedir}/{args.attack}_1_{args.n_ex}_eps_{args.eps:.5f}' +\
                f'_pfyiter_{args.pfy_iter}_dyn_{args.dynamic}_loss_{args.loss}' +\
                f'_{dets}_eot_{args.eot_iter}_niter_{args.n_iter}.pth')
        else:
            torch.save(x_adv, f'{savedir}/eval_fast_{args.data_path}.pth')


if __name__ == '__main__':
    args_ = parse_args()
    main(args_)

