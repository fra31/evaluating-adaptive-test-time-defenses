import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
import argparse
from functools import partial
import os
import math
from datetime import datetime
import time

#import robustbench as rb
from robustbench.utils import clean_accuracy
import autoattack
try:
    import other_utils
except ImportError:
    from autoattack import other_utils

from utils_tta import load_dataset, get_logits, get_wc_acc, eval_fast
from apgd_tta import apgd_restarts

from functions import base_function, defense_function, utils, defense_function_new
from functions.argparse_function import argparser_function


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


@torch.enable_grad()
def purify(x, discr, args, track_interm=False):
    if not args.use_custom_discr:
        x_pfy = defense_function.pgd_defense(discriminator=discr, images=x,
            args=args, track_interm=track_interm)
    else:
        x_pfy = defense_function_new.pgd_defense(discriminator=discr, images=x,
            args=args, track_interm=track_interm)
    return x_pfy


def diff_full_def(x, clf, discr, args):
    z = x.clone().detach()
    with torch.no_grad():
        delta = purify(z, discr, args, track_interm=False) - z
    return clf(x + delta)


def main():
    args = argparser_function() #get_args()

    if not (args.data_path is None or args.data_path in ['multiple']):
        assert os.path.exists(args.data_path)
        x_init = torch.load(args.data_path)
        #other_utils.check_imgs(x_init, x_imgs, args.norm)
        print(x_init.shape)
    else:
        x_init = args.init
    if args.norm == 'Linf':
        args.eps /= 255
    
    device = 'cuda'
    x_test, y_test = load_dataset(args.dataset, args.n_ex, device, args.data_dir)
    
    if isinstance(x_init, torch.Tensor):
        ind = None
    elif not args.attack == 'loss_landscape':
        ind = None
    print(x_test.shape, x_test.max().item(), x_test.min().item(), y_test[:10])
    if isinstance(x_init, torch.Tensor):
        other_utils.check_imgs(x_init, x_test, args.norm)
    
    target_model = base_function.load_target_model(dataset=args.dataset, args=args, device=device)
    discriminator = base_function.load_discriminator(args=args, target_model=target_model, device=device)
    if args.main_classifier == 'madry' and not args.use_rand_discr:
        discriminator.load_state_dict(torch.load(os.path.join(args.pth_path, 'purifier_models',
            args.dataset, args.main_classifier + '.pt'), map_location=device), strict=False)
    else:
        print('using discriminator with random weights')
    discriminator.eval()
    if args.use_custom_discr:
        print('using not original discriminator')
    
    
    diff_aid_defense = partial(diff_full_def, clf=target_model, discr=discriminator, args=args)
    
    test_model = [target_model, diff_aid_defense][0 if args.defense_step == 0 else -1]
    
    savedir = f'./results/{args.dataset}/{args.main_classifier}'
    other_utils.makedir(savedir)
    logsdir = f'./logs/{args.dataset}/{args.main_classifier}'
    other_utils.makedir(logsdir)
    logger = other_utils.Logger(None)

    if args.attack in ['eval_fast', 'eval_fast_short']:
        short_version = 'short' in args.attack
        args.attack = 'eval_fast' # for consistency with older experiments
        runname = f'{args.attack}_1_{args.n_ex}_pursteps_{args.defense_step}' #+ '_basemodel'
        runname += f'{"_short" if short_version else ""}'
        x_adv = eval_fast(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, bs=args.batch_size, short_version=short_version,
            log_path=f'{logsdir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')    
    
    elif args.attack in ['apgd']:
        runname = f'{args.attack}_1_{args.n_ex}_niter_{args.n_iter}_loss_{args.loss}' + \
            f'_nrestarts_{args.n_restarts}_init_{args.init}_pursteps_{args.defense_step}'
        
        #pfy_fn = partial(purify, discr=discriminator, args=args, track_interm=True)
        if isinstance(x_init, torch.Tensor):
            outputs = get_logits(test_model, x_init, bs=args.batch_size)
            pred = outputs.max(1)[1] == y_test
            #x_curr, y_curr = x_test[pred], y_test[pred]
            x_adv = x_init.clone()
            x_init = None
            print(f'initial accuracy={pred.float().mean():.1%}')
            sys.stdout.flush()
        else:
            pred = torch.ones_like(y_test) == torch.ones_like(y_test)
            x_adv = x_test.clone()
            #x_init = None
        x_best = x_adv.clone()
        x_adv_curr, x_best_curr = apgd_restarts(test_model,
            x_test[pred], y_test[pred], n_restarts=args.n_restarts,
            verbose=True, n_iter=args.n_iter, loss=args.loss, eps=args.eps,
            norm=args.norm, eot_iter=0,
            log_path=f'{logsdir}/{runname}.txt')
        x_adv[pred] = x_adv_curr.clone()
        x_best[pred] = x_best_curr.clone()
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        torch.save(x_best, f'{savedir}/{runname}_best.pth')
    
    elif args.attack == 'test_points':
        if not args.data_path is None:
            x_adv = x_init.clone()
        else:
            x_adv = x_test.clone()
            args.data_path = f'original points ({args.n_ex})'
        logger.log_path = '{}/stats_eval.txt'.format(logsdir)
        logger.log(f'purif steps={args.defense_step}')
        logger.log(args.data_path.split('/')[-1])
        str_imgs = other_utils.check_imgs(x_adv, x_test, args.norm)
        logger.log(str_imgs)
    
    
    args.eot_test = 2 # 10 to compute runtime
    for c in range(args.eot_test):
        if c == 1:
            startt = time.time()
        acc = clean_accuracy(test_model, x_adv, y_test, device='cuda',
            batch_size=args.batch_size)
        logger.log(f'robust accuracy={acc:.1%}')
    evalt = time.time() - startt
    try:
        logger.log(runname)
    except:
        pass
    logger.log(f'defense steps={args.defense_step} robust accuracy={acc:.2%}')
    logger.log(f'runs={args.eot_test - 1} single run time={evalt / (args.eot_test - 1):.3f} s')
    logger.log('\n')


if __name__ == '__main__':
    main()


