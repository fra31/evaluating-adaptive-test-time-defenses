import torch
import torch.nn.functional as F
#import robustbench as rb
from robustbench.utils import clean_accuracy, load_model

try:
    import other_utils
except:
    import autoattack.other_utils as other_utils
import autoattack

import math
import sys
import time
import statistics
import copy
sys.path.append('../')
import argparse
import os

from utils_tta import eval_maxloss, get_wc_acc, eval_fast, eval_with_square, \
    get_logits, get_batch, load_dataset
from adaptive_opt import apgd_interm_restarts




'''class HedgeDefense():
    def __init__(self, model, n_cls, eps, n_iter, stepsize):
        assert not model.training
        self.model = model
        self.n_cls = n_cls
        self.n_iter = n_iter
        self.stepsize = stepsize
        self.eps = eps
        #self.loss = lambda x, y: F.cross_entropy(x, y, reduction='none')
        self.track_grad = False
        self.random_step = True
    
    @torch.enable_grad()
    def loss_fn(self, x):
        output = self.model(x)
        loss = torch.zeros(x.shape[0], device=x.device)
        u = torch.ones_like(loss).long()
        for c in range(self.n_cls):
            loss += F.cross_entropy(output, c * u, reduction='none')
        
        return loss

    def __call__(self, x):
        x_adv = x + (torch.rand_like(x) - .5) * 2. * self.eps * float(self.random_step)
        x_adv.requires_grad_(True)
        loss = self.loss_fn(x_adv)
        loss_best = loss.clone().detach()
        x_best = x_adv.clone().detach()
        u = torch.ones_like(x)
        for _ in range(self.n_iter):
            with torch.enable_grad():
                grad = torch.autograd.grad(loss.sum(), x_adv, only_inputs=True,
                    create_graph=self.track_grad)[0]
            #grad.detach_()
            x_adv.detach_()
            x_adv += self.stepsize * grad.sign().detach()
            x_adv = x + torch.min(torch.max(x_adv - x, -self.eps * u), self.eps * u)
            x_adv.clip_(0., 1.)

            x_adv.requires_grad_(True)
            loss = self.loss_fn(x_adv)
            ind = loss > loss_best
            loss_best[ind] = loss[ind].clone().detach()
            x_best[ind] = x_adv[ind].clone().detach()

        return self.model(x_best)'''


class DiffHedgeDefense():
    def __init__(self, model, n_cls, eps, n_iter, stepsize, random_step):
        assert not model.training
        self.model = model
        self.n_cls = n_cls
        self.n_iter = n_iter
        self.stepsize = stepsize
        self.eps = eps
        self.random_step = random_step
        self.return_interm = False
        self.training = self.model.training
    
    def loss_fn(self, x):
        output = self.model(x)
        loss = torch.zeros(x.shape[0], device=x.device)
        u = torch.ones_like(loss).long()
        for c in range(self.n_cls):
            loss += F.cross_entropy(output, c * u, reduction='none')
    
        return loss
    
    @torch.enable_grad()
    def __call__(self, x):
        delta = torch.zeros_like(x) if not self.random_step else (
            torch.rand_like(x) - .5) * 2. * self.eps
        delta.requires_grad_(True)
        u = torch.ones_like(x, requires_grad=False)
        interm_delta = []
        for _ in range(self.n_iter):
            loss = self.loss_fn(x + delta)
            grad = torch.autograd.grad(loss.sum(), delta, retain_graph=False)[0]
            delta.data += self.stepsize * grad.sign()
            delta.data =  delta.data.clip(-self.eps, self.eps) #torch.max(-self.eps * u, torch.min(delta.data, self.eps * u))
            delta.data = (x + delta.data).clip(0., 1.) - x
            interm_delta.append(delta.clone().detach())
        if self.return_interm:
            return (self.model(x + delta), torch.stack(interm_delta, 0
                ) + x.clone().detach().unsqueeze(0))
        return self.model(x + delta)

        





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--n_ex', type=int, default=200)
    parser.add_argument('--model_name', type=str, default='orig_wrn')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--eps', type=float, default=8.)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--data_dir', type=str, default='/home/scratch/datasets/CIFAR10')
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--batch_size', type=int, default=1000)
    # defense parameters
    parser.add_argument('--hd_iter', type=int, default=20)
    parser.add_argument('--hd_rs', action='store_true')
    #parser.add_argument('--eot_def_iter', type=int, default=150)
    # eval parameters
    parser.add_argument('--attack', type=str)
    parser.add_argument('--interm_freq', type=int, default=10)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--loss', type=str, default='ce')
    parser.add_argument('--init', type=str)
    parser.add_argument('--get_trajectory', action='store_true')
    parser.add_argument('--ge_iters', type=int)
    parser.add_argument('--ge_eta', type=float)
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--ge_mu', type=float)
    #parser.add_argument('--lr', type=float)
    parser.add_argument('--use_ls', action='store_true')
    parser.add_argument('--eot_iter', type=int, default=0)
    parser.add_argument('--step_size', type=float)
    parser.add_argument('--zoadamm_lr', type=float, default=.01)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args() #rb.utils.parse_args()

    if not (args.data_path is None or args.data_path in ['multiple']):
        assert os.path.exists(args.data_path)
        x_init = torch.load(args.data_path)
        #other_utils.check_imgs(x_init, x_imgs, args.norm)
        print(x_init.shape)
    else:
        x_init = args.init
    if args.norm == 'Linf':
        args.eps /= 255.
    args.threat_model = args.norm
    
    device = 'cuda:0'
    
    x_test, y_test = load_dataset(args.dataset, args.n_ex, device=device,
        data_dir=args.data_dir)
    
    model = load_model(args.model_name,
                       model_dir=args.model_dir,
                       dataset=args.dataset,
                       threat_model=args.threat_model)
    model.eval()
    model = model.to(device)
    print('model loaded')
    
    savedir = './results/{}/{}/{}'.format(args.dataset, args.threat_model,
        args.model_name)
    other_utils.makedir(savedir)
    logsdir = './logs/{}/{}/{}'.format(args.dataset, args.threat_model,
        args.model_name)
    other_utils.makedir(logsdir)
    #runname = f'{args.attack}_1_{args.n_ex}_niter_{args.n_iter}_args
    logger = other_utils.Logger(None) #'{}/stats_eval.txt'.format(logsdir)
    
    #hd_model = HedgeDefense(model, 10, args.eps, args.hd_iter, 4. / 255.)
    diffhd_model = DiffHedgeDefense(model, 10, args.eps, args.hd_iter,
        4. / 255., args.hd_rs)
    
    
    test_model = model if args.hd_iter == 0 else diffhd_model

    if args.attack == 'eval_fast':
        runname = f'{args.attack}_1_{args.n_ex}_hdsteps_{args.hd_iter}' #+ '_basemodel'
        runname += f'{"_rand" if args.hd_rs else ""}'
        x_adv = eval_fast(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, bs=args.batch_size)
        torch.save(x_adv, f'{savedir}/{runname}.pth')

    elif args.attack in ['interm', 'interm_ge', 'apgd', 'apgd_ge']:
        runname = f'{args.attack}_1_{args.n_ex}_niter_{args.n_iter}_loss_{args.loss}' + \
            f'_nrestarts_{args.n_restarts}_init_{args.init}_pursteps_{args.hd_iter}' + \
            f'{"_rand" if args.hd_rs else ""}'
        #if not ind is None and len(ind) == 1:
        #    runname += f'_{ind[0]}'
        if args.use_ls:
            runname += '_ls'
        if args.eot_iter > 0:
            runname += f'_eotiter_{args.eot_iter}'
        if not args.step_size is None:
            runname += f'_stepsize_{args.step_size:.3f}'
        #pfy_fn = partial(purify, discr=discriminator, args=args, track_interm=True)
        #x_init = None
        x_adv, x_best = [], []
        bs = args.batch_size
        for counter in range(math.ceil(x_test.shape[0] / bs)):
            x_curr, y_curr = get_batch(x_test, y_test, bs, counter, device=device)
            x_adv_curr, x_best_curr = apgd_interm_restarts(test_model, x_curr, y_curr,
                use_interm='interm' in args.attack, stepsize=args.step_size,
                n_restarts=args.n_restarts, verbose=True, n_iter=args.n_iter,
                loss=args.loss, x_init=x_init, eps=args.eps, eot_iter=args.eot_iter,
                norm=args.norm, use_ge='_ge' in args.attack, ge_iters=args.ge_iters,
                ge_eta=args.ge_eta, ge_prior=args.use_prior, ge_mu=args.ge_mu,
                #pfy_fn=pfy_fn, clf=target_model,
                use_ls=args.use_ls, #'_ge' in args.attack
                log_path=f'{savedir}/{runname}.txt')
            x_adv.append(x_adv_curr), x_best.append(x_best_curr)
        x_adv, x_best = torch.cat(x_adv, 0), torch.cat(x_best, 0)
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        torch.save(x_best, f'{savedir}/{runname}_best.pth')

    elif args.attack == 'test_points':
        if not args.data_path is None:
            x_adv = x_init.clone()
        else:
            x_adv = x_test.clone()
            args.data_path = f'original points ({args.n_ex})'
        str_imgs = other_utils.check_imgs(x_adv.cpu(), x_test.cpu(), args.norm)
        logger.log_path = '{}/stats_eval.txt'.format(logsdir)
        logger.log(f'[hd] steps={args.hd_iter} rand step={args.hd_rs}')
        logger.log(args.data_path.split('/')[-1])
        logger.log(str_imgs)
    
    elif args.attack == 'loss_landscape':
        from utils_tta import get_2D_losslandscape
        ind = [7,  16,  18]
        x_test, y_test = x_test[ind], y_test[ind]
        if not ind is None:
            #ind = ind[:3]
            ind = '-'.join([f'{c:.0f}' for c in ind])
        print(ind)
        if isinstance(x_init, torch.Tensor):
            x_test = x_init
        eta, vals = get_2D_losslandscape(test_model, x_test, y_test,
            delta=args.eps / 10., loss=args.loss)
        torch.save({'eta': eta, 'vals': vals}, f'{savedir}/pl_losslandscape_{ind}' + \
            f'_pursteps_{args.defense_step}_loss_{args.loss}' + \
            f'{"_randdiscr" if args.use_rand_discr else ""}.pth')
        sys.exit()
    
    elif args.attack == 'find_wc':
        logger.log_path = '{}/stats_eval.txt'.format(logsdir)
        logger.log(f'[hd] steps={args.hd_iter} rand step={args.hd_rs}')
        xs = [f'apgd_1_{args.n_ex}_niter_50_loss_dlr-targeted_nrestarts_5_init_None_pursteps_5_rand_eotiter_3_best.pth',
            f'apgd_1_{args.n_ex}_niter_50_loss_dlr-targeted_nrestarts_5_init_None_pursteps_0_best.pth'
            ]
        for x in xs:
            assert os.path.exists(f'{savedir}/{x}'), f'missing {x}'
        logger.log('\n'.join(xs))
        xs = [torch.load(f'{savedir}/{x}').cpu() for x in xs]
        with torch.no_grad():
            acc, x_adv = get_wc_acc(test_model, xs, y_test,
                bs=args.batch_size, device=device,
                eot_test=1 + 4 * int(args.hd_rs))
        str_imgs = other_utils.check_imgs(x_adv.cpu(), x_test.cpu(), args.norm)
        runname = f'wc_multiple_1_{args.n_ex}_pursteps_{args.hd_iter}'
        runname += f'{"_rand" if args.hd_rs else ""}'
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        logger.log(runname)
        logger.log(str_imgs)
    
    
    l_acc = []
    #logger.log(runname)
    #logger.log(str_imgs)
    #startt = time.time()
    for c in range(10 + 0 * int(args.hd_rs)):
        if c == 1:
            startt = time.time()
        acc = clean_accuracy([test_model, #hd_model
            ][0], x_adv, y_test, device='cuda',
            batch_size=args.batch_size)
        logger.log(f'defense steps={args.hd_iter} robust accuracy={acc:.1%}')
        l_acc.append(acc)
    evalt = time.time() - startt
    logger.log(f'defense steps={args.hd_iter} runs={len(l_acc)}' + \
        f' robust accuracy={statistics.mean(l_acc):.1%} ({statistics.pstdev(l_acc) * 100:.2f})' + \
        f' single run time={evalt / (len(l_acc) - 1):.3f} s')
    logger.log('\n')

