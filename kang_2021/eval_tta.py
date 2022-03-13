import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
#sys.path.append('../../')
import argparse
import os
from functools import partial
import time
import statistics

from robustbench.utils import clean_accuracy, load_model
import autoattack
try:
    import other_utils
except ImportError:
    from autoattack import other_utils
from utils_tta import load_dataset, eval_fast, get_wc_acc, parse_args
from apgd_tta import apgd_restarts

#


def load_models(modelname, device, modeldir, modeldir_rb):
    if modelname == 'rebuffi_orig':
        model = load_model('Rebuffi2021Fixing_70_16_cutmix_extra',
            model_dir=modeldir_rb)
        model.to(device)
        model.eval()
        return model
        
    elif modelname == 'rebuffi_sodef':
        from models import create_model
        new_model = create_model(device, modeldir_rb)
        ckpt = torch.load(f'{modeldir}/Rebuffi2021Fixing_70_16_cutmix_extra/full.pth')
        new_model.load_state_dict(ckpt['state_dict'])
        new_model.eval()
        return new_model
        
    elif modelname == 'trades':
        from wideresnet import WideResNet
        model = WideResNet()
        ckpt = torch.load(f'{modeldir}/EXP/pre_train_model/wide10_trades_eps8_tricks.pt')
        model.load_state_dict(ckpt)
        model.to(device)
        model.eval()
        return model
        
    elif modelname == 'trades_sodef':
        from models_small import create_model
        model = create_model(device)
        ckpt = torch.load(f'{modeldir}/EXP/nips_model/full.pth')
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model
        
    
class ModelWithFeatures(nn.Module):
    """ to use both logits and intermediate features
    """
    def __init__(self, composite_model):
        super().__init__()
        self.composite_model = composite_model
        
    def forward(self, x, return_fts=False):
        if not return_fts:
            return self.composite_model(x)
            
        else:
            fts = self.composite_model[0](x)
            out = self.composite_model[1](fts)
            out = self.composite_model[2](out)
            return out, fts


if __name__ == '__main__':
    args = parse_args()
    device = 'cuda'
    
    if args.eps > 1. and args.norm == 'Linf':
        args.eps /= 255.
    if not args.data_path is None:
        assert os.path.exists(args.data_path)
    
    x_test, y_test = load_dataset(args.dataset, args.n_ex, device=device,
        data_dir=args.data_dir)
    print('data loaded')

    model = load_models(args.model, device, args.model_dir, args.model_dir_rb)
    print('model loaded')

    logger = other_utils.Logger(None)
    savedir = f'./results/{args.model}'
    other_utils.makedir(savedir)
    logsdir = f'./logs/{args.model}'
    other_utils.makedir(logsdir)
    logger.log_path = f'{logsdir}/stats.txt'

    if args.loss in ['l2', 'l1', 'linf']:
        assert 'sodef' in args.model
        test_model = ModelWithFeatures(model)
        test_model.eval()
    else:
        test_model = model
    
    runname = f'{args.attack}_1_{args.n_ex}_{args.norm}_eps_{args.eps:.5f}'
    
    if not args.indices is None:
        notes_run = f'_{args.indices}'
        ind = args.indices.split('-')
        ind_restr = list(range(int(ind[0]), int(ind[1])))
        print(ind_restr)
        sys.stdout.flush()
        x_test, y_test = x_test[ind_restr], y_test[ind_restr]
        runname += notes_run

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
            log_path=f'{logsdir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        
    elif args.attack in ['interm', 'interm_ge', 'apgd', 'apgd_ge', 'apgd_on_features']:
        runname += f'_niter_{args.n_iter}_loss_{args.loss}' + \
            f'_nrestarts_{args.n_restarts}_init_{args.init}'
        if not args.step_size is None:
            runname += f'_stepsize_{args.step_size:.3f}'
        x_init = None
        n_batches = math.ceil(args.n_ex / args.batch_size)
        x_adv, x_best = x_test.clone(), x_test.clone()
        for c in range(n_batches):
            x_test_curr = x_test[c * args.batch_size:(c + 1) * args.batch_size]
            y_test_curr = y_test[c * args.batch_size:(c + 1) * args.batch_size]
            x_adv_curr, x_best_curr = apgd_restarts(test_model,
                x_test_curr, y_test_curr, n_restarts=args.n_restarts,
                verbose=True, n_iter=args.n_iter, loss=args.loss, eps=args.eps,
                norm=args.norm, log_path=f'{logsdir}/{runname}.txt')
            x_adv[c * args.batch_size:(c + 1) * args.batch_size] = x_adv_curr.clone()
            x_best[c * args.batch_size:(c + 1) * args.batch_size] = x_best_curr.clone()
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
            # feature attacks
            'apgd_1_1000_Linf_eps_0.03137_niter_100_loss_l1_nrestarts_1_init_None.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_25_loss_l1_nrestarts_4_init_None.pth',
            #
            # AA on the original model
            'eval_fast_1_1000_Linf_eps_0.03137.pth',
            #
            #'eval_with_square_1_1000_Linf_eps_0.03137_niter_5000_loss_ce_nrestarts_1.pth'
            ]
        xs = [f'./results/rebuffi_sodef/{c}' for c in xs]
        for x in xs:
            assert os.path.exists(x), f'missing {x}'
        xtr = [
            # transfer apgd (points with best loss for unsuccessful cases are used)
            'apgd_1_1000_Linf_eps_0.03137_niter_100_loss_dlr-targeted_nrestarts_9_init_None_best.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_99_loss_cw_nrestarts_1_init_None_best.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_99_loss_dlr-targeted_nrestarts_3_init_None_best.pth',
            'apgd_1_1000_Linf_eps_0.03137_niter_99_loss_ce_nrestarts_1_init_None_best.pth',
            ]
        xtr = [c for c in xtr if args.norm in c]
        xtr = [f'./results/rebuffi_orig/{c}' for c in xtr]
        for x in xtr:
            assert os.path.exists(x), f'missing {x}'
        logger.log('\n'.join(xs))
        xs = [torch.load(x).cpu() for x in xs]
        logger.log('\n'.join(xtr))
        xs += [torch.load(x).cpu() for x in xtr]
        xs = xs[::-1]
        for i, x in enumerate(xs):
            other_utils.check_imgs(xs[i].cpu(), x_test.cpu(), args.norm)
        with torch.no_grad():
            acc, x_adv = get_wc_acc(model, xs, y_test,
                bs=args.batch_size, device=device, eot_test=1, logger=logger,
                loss=args.loss)
        #torch.save(x_adv, f'{savedir}/{runname}.pth')

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
        logger.log(f'[{args.model}] robust acc={acc:.1%}')
    try:
        evalt = time.time() - startt
    except:
        evalt = 0.
    logger.log(f'runs={args.eot_test - 1} single run time={evalt / (args.eot_test - 1):.3f} s')
    logger.log('')

