import torch
import torch.nn.functional as F
import sys
sys.path.append('../')
import argparse
import os
import yaml
from functools import partial
import time
import statistics
from datetime import datetime

#import robustbench as rb
from robustbench.utils import load_model
from robustbench.utils import clean_accuracy
import autoattack
import other_utils
from utils_tta import load_dataset, get_logits, get_wc_acc, eval_fast, \
    get_2D_losslandscape, clean_acc_with_eot, eval_maxloss, eval_with_square, \
    clf_with_eot, get_wc_doubleeot
from adaptive_opt import apgd_interm_restarts, apgd_twomodels

from ncsnv2.runners.ncsn_runner import get_model
from purification.adp import adp
from utils.accuracy import gen_ll
from utils.transforms import raw_to_ebm, ebm_to_raw


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # Dataset and save logs
    #parser.add_argument('--log', default='imgs', help='Output path, including images and logs')
    parser.add_argument('--config', type=str, default='default.yml',  help='Path for saving running related data.')
    
    # added
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
    parser.add_argument('--indices', type=str)
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
        new_config = dict2namespace(config)
    
    return args, new_config

    
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_ebm(ebm_path, ebm_config_path, device='cuda'):
    with open(ebm_config_path, 'r') as f:
        config_ebm_orig = yaml.load(f, Loader=yaml.Loader)
        config_ebm = dict2namespace(config_ebm_orig)
        config_ebm.device = device
    network_ebm = get_model(config_ebm)
    network_ebm = torch.nn.DataParallel(network_ebm)
    states_ebm = torch.load(ebm_path, map_location=device) #os.path.join('ncsnv2/run/logs', self.config.structure.ebm_log, 'checkpoint.pth'), map_location=self.config.device.ebm_device
    network_ebm.load_state_dict(states_ebm[0], strict=True)
    print('ebm loaded')
    return network_ebm
    

""" as close as possible to original implementation """
def pfy(x, network_ebm, config):
    x_pur_list_list = []
    step_size_list_list = []
    #start_time = datetime.now()
    #print("[{}] Epoch {}:\tBegin purifying {} attacked images".format(str(datetime.now()), i, self.config.structure.bsize))
    for j in range(config.purification.rand_smoothing_ensemble):
        if config.purification.purify_method=="adp_multiple_noise":
            if j < config.purification.rand_smoothing_ensemble/2:
                config.purification.rand_smoothing_level = config.purification.lowest_level
            else:
                config.purification.rand_smoothing_level = config.purification.highest_level
            x_pur_list, step_size_list = adp(x, network_ebm, config.purification.max_iter, mode="purification", config=config)
        else:
            #x_pur_list, step_size_list = eval(config.purification.purify_method)(x_adv, network_ebm, config.purification.max_iter, mode="purification", config=config)
            x_pur_list, step_size_list = adp(x, network_ebm, config.purification.max_iter, mode="purification", config=config)
        x_pur_list_list.append(x_pur_list)
        step_size_list_list.append(step_size_list)
    return x_pur_list_list, step_size_list_list

    
def output_final_step(truth_ll): #acc_final_step(truth_ll, ground_label)
    # output: final_correct, correct
    total_logit = torch.zeros_like(truth_ll["logit"][0][0])
    total_softmax = torch.zeros_like(truth_ll["softmax"][0][0])
    total_onehot = torch.zeros_like(truth_ll["softmax"][0][0]) # NOT typo, "softmax" is right
    list_noisy_inputs_logit = []
    list_noisy_inputs_softmax = []
    list_noisy_inputs_onehot = []
    # accuracy, involving final step
    for i in range(len(truth_ll["logit"])): # list of list of [bsize, nClass]
        total_logit += truth_ll["logit"][i][-1]
        #list_noisy_inputs_logit.append(torch.eq(torch.argmax(total_logit, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["softmax"])): # list of list of [bsize, nClass]
        total_softmax += truth_ll["softmax"][i][-1]
        #list_noisy_inputs_softmax.append(torch.eq(torch.argmax(total_softmax, dim=1), ground_label).sum().float().to('cpu').numpy())
    for i in range(len(truth_ll["onehot"])): # list of list of [bsize]
        for k in range(truth_ll["onehot"][i][0].shape[0]):
            total_onehot[k][truth_ll["onehot"][i][-1][k]] += 1.
        #list_noisy_inputs_onehot.append(torch.eq(torch.argmax(total_onehot, dim=1), ground_label).sum().float().to('cpu').numpy())
    return total_logit, total_softmax, total_onehot
    

def clf_purified_imgs(x_pur_list_list, network_clf, config):
    transform_raw_to_clf = lambda x: x
    att_list_list_dict = gen_ll(x_pur_list_list, network_clf, transform_raw_to_clf, config)
    logits, softmaxs, onehots = output_final_step(att_list_list_dict)
    return logits, softmaxs
    

@torch.enable_grad()
def adp_defense(x, clf, ebm, config):
    z = x.clone()
    x_pfy, _ = pfy(z, ebm, config)
    logits, softmaxs = clf_purified_imgs(x_pfy, clf, config)
    return softmaxs
    

""" reimplementation for custom eval """
@torch.enable_grad()
def pfy_single(x, network_ebm, config):
    x_pur_list, step_size_list = adp(x, network_ebm, config.purification.max_iter, mode="purification", config=config)
    return x_pur_list[-1], x_pur_list

    
def adp_single(x, clf, ebm, config, return_interm=False):
    z = x.clone()
    x_pfy, interm_x = pfy_single(z, ebm, config)
    #logits = get_logits(clf, x_pfy, bs=bs, track
    delta = x_pfy.detach() - x.clone().detach()
    logits = clf(x + delta)
    return logits


""" reimplementation to allow gradient prop. """
def diff_adp(x, network_ebm, clf, config, only_pur=False):
    min_step_lr = 0.00001
    lr_min = 1.0e-3
    images = [] # From noisy initialized image to purified image
    step_sizes = [] # Step sizes

    transform_raw_to_ebm = [raw_to_ebm(config.structure.dataset), lambda x: x][0]
    transform_ebm_to_raw = [ebm_to_raw(config.structure.dataset), lambda x: x][0]
    
    smoothing_level = config.purification.rand_smoothing_level
    #x_pur = torch.clamp(x + torch.randn_like(x) * smoothing_level, 0.0, 1.0)
    with torch.no_grad():
        init_noise = (x + torch.randn_like(x) * smoothing_level).clamp(0., 1.) - x
    x_pur = x + init_noise
    labels = torch.ones(x.shape[0], device=x.device).long()
    step_lambda = config.purification.purification_lambda
    max_iter = config.purification.max_iter
    
    cont_purification = torch.ones(x_pur.shape[0], dtype=torch.bool).to(config.device.ebm_device)
    # Stopping criterion
    for i in range(max_iter):
        #labels = torch.ones(x_pur.shape[0], device=x_pur.device)
        #labels = labels.long().to(config.device.ebm_device)
        grad = network_ebm(x_pur, labels) # Get gradients
        # Get adaptive step size
        with torch.no_grad():
            x_eps = x_pur + lr_min*grad
            grad_eps = network_ebm(x_eps, labels)
            z1 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad_eps.view(grad_eps.shape[0], -1, 1))
            z2 = torch.bmm(grad.view(grad.shape[0], 1, -1), grad.view(grad.shape[0], -1, 1))
            z = torch.div(z1, z2)
            step_size = torch.clamp(step_lambda*lr_min/(1.-z), min=min_step_lr, max=min_step_lr*10000.).view(-1)
            cont_purification = torch.logical_and(cont_purification, (step_size>config.purification.stopping_alpha))
            if torch.sum(cont_purification)==0:
                break
            step_size *= cont_purification
        #x_pur_t = x_pur.clone().detach()
        #x_pur = torch.clamp(transform_ebm_to_raw(x_pur_t+grad*step_size[:, None, None, None]), 0.0, 1.0)
        x_pur = transform_ebm_to_raw(x_pur + grad * step_size.view(-1, 1, 1, 1))
        x_pur.clamp_(0., 1.)
        
    if only_pur:
        return x_pur
    
    return clf(x_pur)
    

if __name__ == '__main__':
    args, config = parse_args_and_config()
    device = 'cuda'
    #
    #
    if args.eot_def_iter is None:
        args.eot_def_iter = 10
    print(f'using defense with {args.eot_def_iter} runs')
    if not args.sigma_def is None:
        config.purification.rand_smoothing_level = args.sigma_def
    print(f'use rand={config.purification.rand_smoothing} sigma={config.purification.rand_smoothing_level}')
    if not args.max_iters_def is None:
        config.purification.max_iter = args.max_iters_def
    print(f'using defense with {config.purification.max_iter} iters')
    # custom because missing
    config.purification.rand_type = 'non-binary'
    
    if not args.seed is None:
        torch.manual_seed(args.seed)
    if args.eps > 1. and args.norm == 'Linf':
        args.eps /= 255.
    if not args.data_path is None:
        assert os.path.exists(args.data_path)
    
    x_test, y_test = load_dataset(args.dataset, args.n_ex, device=device,
        data_dir=args.data_dir)
        
    if not args.indices is None:
        notes_run = f'_{args.indices}'
        ind = args.indices.split('-')
        ind_restr = list(range(int(ind[0]), int(ind[1])))
        #
        print(ind_restr)
        sys.stdout.flush()
        x_test, y_test = x_test[ind_restr], y_test[ind_restr]
    
    # load models
    ebm_path = './ncsnv2/exp/logs/cifar10/best_checkpoint_with_denoising.pth' #"/mnt/SHARED/fcroce42/test_time_defenses_eval/adp/ncsnv2/exp/logs/cifar10/best_checkpoint_with_denoising.pth"
    ebm_config_path = './ncsnv2/configs/cifar10.yml'
    ebm = load_ebm(ebm_path, ebm_config_path, device)
    print(ebm.training)
    ebm.eval()
    print(ebm.training)
    
    model = load_model(args.model,
                       model_dir=args.model_dir,
                       dataset=args.dataset,
                       threat_model=args.norm)
    model.eval()
    model = model.to(device)
    print('clf loaded')

    logger = other_utils.Logger(None)
    savedir = f'./results/{args.model}' #{args.attack}_eotiter_{args.eot_def_iter}_k_{args.lng_steps}
    other_utils.makedir(savedir)
    logsdir = f'./logs/{args.model}'
    other_utils.makedir(logsdir)
    logger.log_path = f'{logsdir}/stats.txt'
    
    

    
    adp_fn = partial(adp_single, clf=model, ebm=ebm, config=config) # original, without eot
    #adp_fn_multiple = partial(clf_with_eot, model=adp_fn, eot_test=args.eot_test)
    diff_adp_fn = partial(diff_adp, network_ebm=ebm, clf=model, config=config) # reimplemented in a differentiable way
    
    
    
    if args.only_clean:
        sys.exit()
    
    
    
    # select which version is used in the attacks
    test_model = [adp_fn, diff_adp_fn][-1]

    if args.attack == 'eval_fast':
        short_version = True
        runname = f'{args.attack}_1_{args.n_ex}_eotiter_{args.eot_test}' #pursteps_{args.defense_step} #+ '_basemodel'
        runname += f'{"_short" if short_version else ""}'
        x_adv = eval_fast(test_model, x_test, y_test, eps=args.eps,
            savedir=savedir, bs=args.batch_size, short_version=short_version,
            eot_iter=args.eot_test,
            log_path=f'{savedir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        
    elif args.attack in ['interm', 'interm_ge', 'apgd', 'apgd_ge']:
        runname = f'{args.attack}_1_{args.n_ex}_niter_{args.n_iter}_loss_{args.loss}' + \
            f'_nrestarts_{args.n_restarts}_init_{args.init}' #_pursteps_{args.defense_step}
        runname += f'_sigma_{config.purification.rand_smoothing_level}'
        runname += f'_maxiter_{config.purification.max_iter}'
        if args.eot_test > 0:
            runname += f'_eotiter_{args.eot_test}'
        if not args.step_size is None:
            runname += f'_stepsize_{args.step_size:.3f}'
        if not args.indices is None:
            runname += f'_{args.indices}'
        logger.log_path = f'{logsdir}/{runname}.txt'
        #pfy_fn = partial(purify, discr=discriminator, args=args, track_interm=True)
        pfy_single_fn = partial(pfy_single, network_ebm=ebm, config=config)
        x_init = None
        if not args.data_path is None:
            x_init = torch.load(args.data_path)
        x_adv, x_best = apgd_interm_restarts(test_model if not '_ge' in args.attack else adp_fn_multiple,
            model, [pfy_single_fn, diff_adp_only_fn][-1], x_test,
            y_test, use_interm='interm' in args.attack, step_size=args.step_size,
            n_restarts=args.n_restarts, verbose=True, n_iter=args.n_iter,
            loss=args.loss, ebm=[None, ebm][-1], x_init=x_init, eps=args.eps,
            norm=args.norm, use_ge='_ge' in args.attack, ge_iters=args.ge_iters,
            ge_eta=args.ge_eta, ge_prior=args.use_prior, bpda_type=None, #args.bpda_type
            eot_test=args.eot_test,
            log_path=f'{logsdir}/{runname}.txt')
        torch.save(x_adv, f'{savedir}/{runname}.pth')
        torch.save(x_best, f'{savedir}/{runname}_best.pth')
        
    elif args.attack == 'test_points':
        if not args.data_path is None:
            runname = f'{args.attack} {args.data_path.split("/")[-1]}'
            x_adv = torch.load(args.data_path).to(device)
        else:
            runname = f'{args.attack} original points ({args.n_ex})'
            x_adv = x_test.clone()
            
    
            
    if False:
        # only to compute runtime
        l_acc = []
        logger.log(runname)
        logger.log(f'bs eval={args.batch_size}')
        startt = time.time()
        for c in range(args.eot_test):
            acc = clean_accuracy(model, x_adv, y_test, batch_size=args.batch_size)
            logger.log(f'acc={acc:.1%}')
            l_acc.append(acc)
        evalt = time.time() - startt
        logger.log('[base model] ' + \
            f' robust accuracy={statistics.mean(l_acc):.1%} ({statistics.pstdev(l_acc) * 100:.2f})' + \
            f' single run time={evalt / len(l_acc):.3f} s')
        logger.log('\n')
        sys.exit()
    
        
    l_acc = []
    logger.log(runname)
    logger.log(f'bs eval={args.batch_size}')
    startt = time.time()
    for c in range(min(args.eot_test, 10)):
        acc, acc_dets = clean_acc_with_eot(adp_fn, x_adv, y_test, bs=args.batch_size,
            eot_test=args.eot_def_iter, method='softmax', return_acc_dets=True)
        if c == 0:
            str_imgs = other_utils.check_imgs(x_adv, x_test, norm=args.norm)
            logger.log(str_imgs)
        acc_wc = (acc_dets == args.eot_def_iter).float().mean()
        logger.log(f'robust accuracy={acc.cpu().float().mean():.1%} (wc={acc_wc:.1%})')
        l_acc.append(acc.cpu().float().mean().item())
    
        try:
            # if attack returns also points with highest loss
            acc, acc_dets = clean_acc_with_eot(adp_fn, x_best, y_test, bs=args.batch_size,
                eot_test=args.eot_def_iter, method='softmax', return_acc_dets=True)
            if c == 0:
                str_imgs = other_utils.check_imgs(x_best, x_test, norm=args.norm)
                logger.log(str_imgs)
            acc_wc = (acc_dets == args.eot_def_iter).float().mean()
            logger.log(f'robust accuracy (best loss)={acc.cpu().float().mean():.1%} (wc={acc_wc:.1%})')
        except:
            pass
    evalt = time.time() - startt
    logger.log(f'[adp] eot iter={args.eot_def_iter} maxiter={config.purification.max_iter}' + \
        f' sigma={config.purification.rand_smoothing_level} runs={len(l_acc)}' + \
        f' robust accuracy={statistics.mean(l_acc):.1%} ({statistics.pstdev(l_acc) * 100:.2f})' + \
        f' single run time={evalt / len(l_acc):.3f} s' + \
        '')
    logger.log('\n')


