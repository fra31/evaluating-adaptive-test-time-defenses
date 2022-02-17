import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../../')
import argparse
import os
#import yaml
from functools import partial
import time
import statistics

import robustbench as rb
from robustbench.utils import clean_accuracy
import autoattack
import other_utils
from utils_tta import load_dataset, eval_fast, eval_maxloss, eval_with_square, \
    get_wc_acc, get_logits
from adaptive_opt import apgd_interm_restarts

from wideresnet import WideResNet_save, WideResNet
from torchdiffeq import odeint_adjoint as odeint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--n_ex', type=int, default=200)
    parser.add_argument('--model', type=str, default='orig_wrn')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--eps', type=float, default=8.)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int, default=100)
    # defense parameters
    parser.add_argument('--max_iters_def', type=int)
    parser.add_argument('--sigma_def', type=float)
    parser.add_argument('--eot_def_iter', type=int)
    # eval parameters
    parser.add_argument('--attack', type=str)
    parser.add_argument('--interm_freq', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--init', type=str)
    parser.add_argument('--eot_test', type=int, default=1)
    parser.add_argument('--ge_iters', type=int)
    parser.add_argument('--ge_eta', type=float)
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--use_ls', action='store_true')
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--only_clean', action='store_true')
    
    args = parser.parse_args()
    return args
    

""" from original implementation """
endtime = 5
fc_dim = 64
act = torch.sin 
f_coeffi = -1
layernum = 0
tol = 1e-3


class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


class ODEfunc_mlp(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 256)
        self.act1 = act
        self.fc2 = ConcatFC(256, 256)
        self.act2 = act
        self.fc3 = ConcatFC(256, 64)
        self.act3 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = f_coeffi*self.fc1(t, x)
        out = self.act1(out)
        out = f_coeffi*self.fc2(t, out)
        out = self.act2(out)
        out = f_coeffi*self.fc3(t, out)
        out = self.act3(out)
        
        return out

    
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

    
class MLP_OUT_Linear(nn.Module):

    def __init__(self):
        super(MLP_OUT_Linear, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)
    def forward(self, input_):
#         h1 = F.relu(self.fc0(input_))
        h1 = self.fc0(input_)
        return h1



if __name__ == '__main__':
    args = parse_args()
    device = 'cuda'
    
    if args.eps > 1. and args.norm == 'Linf':
        args.eps /= 255.
    if not args.data_path is None:
        assert os.path.exists(args.data_path)
    
    x_test, y_test = load_dataset(args.dataset, args.n_ex, device=device,
        data_dir=args.data_dir)

    if args.model == 'orig_wrn':
        model = WideResNet_save(fc_dim, 34, 10, widen_factor=10, dropRate=0.0)
        model.eval()
        odefunc = ODEfunc_mlp(0)
        odefunc.eval()
        feature_layers = [ODEBlock(odefunc)] 
        feature_layers[0].eval()
        fc_layers = [MLP_OUT_Linear()]
        fc_layers[0].eval()
        model = nn.Sequential(model, *feature_layers, *fc_layers).to(device)
        ckpt = torch.load('./EXP/nips_model/full.pth')
        model.load_state_dict(ckpt['state_dict'])
    elif args.model == 'wrn_at':
        model = WideResNet()
        ckpt = torch.load('./EXP/pre_train_model/wide10_trades_eps8_tricks.pt')
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print('model loaded')
    
    logger = other_utils.Logger(None)
    savedir = f'./results/{args.model}'
    other_utils.makedir(savedir)
    logsdir = f'./logs/{args.model}'
    other_utils.makedir(logsdir)
    logger.log_path = f'{logsdir}/stats.txt'

    x = torch.rand([10, 3, 32, 32], device=device)
    x = x_test[:10]
    for _ in range(1):
        with torch.no_grad():
            output = model(x)
        print(output.shape, output)
    if False:
        bs = 256
        for bs in [10, 50]: #range(1)
            acc = clean_accuracy(model, x_test, y_test, device=device,
                batch_size=bs)
            print(f'bs={bs} clean accuracy={acc:.2%}')
    if args.only_clean:
        sys.exit()

    test_model = model
    
    runname = f'{args.attack}_1_{args.n_ex}_{args.norm}_eps_{args.eps:.5f}'

    if args.attack == 'eval_fast':
        short_version = False #True
        #runname = f'{args.attack}_1_{args.n_ex}'
        runname += f'{"_short" if short_version else ""}'
        #runname += f'_{args.norm}_eps_{args.eps:.5f}'
        x_adv = eval_fast(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, bs=args.batch_size, short_version=short_version,
            eot_iter=args.eot_test, norm=args.norm,
            log_path=f'{savedir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        
    elif args.attack == 'maxloss':
        runname += f'_niter_{args.n_iter}_loss_{args.loss}'
        x_adv = eval_maxloss(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, #eot_iter=args.eot_test,
            verbose=True, seed=args.seed, norm=args.norm,
            bs=args.batch_size, loss=args.loss, n_iter=args.n_iter,
            log_path=f'{savedir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        
    elif args.attack == 'eval_with_square':
        runname += f'_niter_{args.n_iter}_loss_{args.loss}'
        runname += f'_nrestarts_{args.n_restarts}'
        x_adv = eval_with_square(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, #eot_iter=args.eot_test, #verbose=Truer
            seed=args.seed, opt_loss=True, n_restarts=args.n_restarts,
            bs=args.batch_size, loss=args.loss, n_iter=args.n_iter, norm=args.norm,
            log_path=f'{savedir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        
    elif args.attack in ['interm', 'interm_ge', 'apgd', 'apgd_ge']:
        runname += f'_niter_{args.n_iter}_loss_{args.loss}' + \
            f'_nrestarts_{args.n_restarts}_init_{args.init}' #_pursteps_{args.defense_step}
        if not args.step_size is None:
            runname += f'_stepsize_{args.step_size:.3f}'
        x_init = None
        x_adv, x_best = apgd_interm_restarts(test_model, test_model, None, x_test,
            y_test, use_interm='interm' in args.attack, #step_size=args.step_size,
            n_restarts=args.n_restarts, verbose=True, n_iter=args.n_iter,
            loss=args.loss, ebm=None, x_init=x_init, eps=args.eps,
            norm=args.norm, use_ge='_ge' in args.attack, ge_iters=args.ge_iters,
            ge_eta=args.ge_eta, ge_prior=args.use_prior, bpda_type=None, #args.bpda_type
            eot_test=0, #args.eot_test
            log_path=f'{logsdir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        torch.save(x_best, f'{savedir}/{runname}_best.pth')
        
    elif args.attack == 'test_points':
        if not args.data_path is None:
            runname += f' {args.data_path.split("/")[-1]}'
            x_adv = torch.load(args.data_path)
        else:
            runname += ' clean points'
            x_adv = x_test.clone()
        print(x_adv.shape)
        
    elif args.attack in 'find_wc':
        xs = [
            ]
        for x in xs:
            assert os.path.exists(f'{savedir}/{x}'), f'missing {x}'
        xtr = ['apgd_1_1000_L2_eps_0.50000_niter_100_loss_ce_nrestarts_1_init_None_best.pth',
            'apgd_1_1000_L2_eps_0.50000_niter_100_loss_cw_nrestarts_1_init_None_best.pth',
            'apgd_1_1000_L2_eps_0.50000_niter_100_loss_dlr-targeted_nrestarts_3_init_None_best.pth',
            #
            'apgd_1_1000_Linf_eps_0.03137_niter_100_loss_ce_nrestarts_1_init_None_best.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_100_loss_cw_nrestarts_1_init_None_best.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_100_loss_dlr-targeted_nrestarts_3_init_None_best.pth'
            ]
        xtr = [c for c in xtr if args.norm in c]
        xtr = [f'./results/wrn_at/{c}' for c in xtr]
        for x in xtr:
            assert os.path.exists(x), f'missing {x}'
        logger.log('\n'.join(xs))
        xs = [torch.load(f'{savedir}/{x}').cpu() for x in xs]
        logger.log('\n'.join(xtr))
        xs += [torch.load(x).cpu() for x in xtr]
        for i, x in enumerate(xs):
            if x.shape[0] < x_test.shape[0]:
                x_new = x_test.clone().to(x.device)
                x_new[:x.shape[0]] = x.clone()
                xs[i] = x_new.clone()
                print(x.shape, x_new.shape)
            other_utils.check_imgs(xs[i].cpu(), x_test.cpu(), args.norm)
        with torch.no_grad():
            acc, x_adv = get_wc_acc(model, xs, y_test,
                bs=args.n_ex, device=device, eot_test=1)
        torch.save(x_adv, f'{savedir}/{runname}.pth')

    l_acc = []
    logger.log(runname)
    for c in range(args.eot_test):
        if c == 1:
            startt = time.time() # skip first since it's usually slower
        acc = clean_accuracy(model, x_adv, y_test, device=device,
            batch_size=args.batch_size)
        if c == 0:
            str_imgs = other_utils.check_imgs(x_adv.to(x_test.device), x_test, norm=args.norm)
            logger.log(str_imgs)
        logger.log(f'[sodef] robust acc={acc:.1%}')
    evalt = time.time() - startt
    logger.log(f'runs={args.eot_test - 1} single run time={evalt / (args.eot_test - 1):.3f} s')
    logger.log('')


