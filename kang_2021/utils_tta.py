import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import argparse
import math
from typing import Tuple

import autoattack
try:
    import other_utils
except ImportError:
    from autoattack import other_utils
import robustbench as rb

from apgd_tta import criterion_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--n_ex', type=int, default=200)
    parser.add_argument('--model', type=str, default='orig_wrn')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--eps', type=float, default=8.)
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--model_dir', type=str, default='./models')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--model_dir_rb', type=str, default='./models')
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
    return args


def load_dataset(dataset: str, n_ex: int = 1000, device: str = 'cuda',
    data_dir: str = None #'/home/scratch/datasets/CIFAR10'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if data_dir is None:
        data_dir = f'/home/scratch/datasets/{dataset.upper()}'
    prepr = transforms.Compose([transforms.ToTensor()])
    
    if dataset in ['cifar10', 'cifar100']:
        dataset_: rb.model_zoo.enums.BenchmarkDataset = rb.model_zoo.enums.BenchmarkDataset(dataset)
        #threat_model_: rb.model_zoo.enums.ThreatModel = rb.model_zoo.enums.ThreatModel(args.threat_model)
        
        x_test, y_test = rb.data.load_clean_dataset(dataset_, n_ex, data_dir,
            prepr=prepr)
    elif dataset == 'svhn':
        dataset = datasets.SVHN(root=data_dir,
                                split='test',
                                transform=transforms.Compose([transforms.ToTensor()]),
                                download=True)
        x_test, y_test = rb.data._load_dataset(dataset, n_ex)
    
    return x_test.to(device), y_test.to(device)


def get_logits(model, x_test, bs=1000, device=None, n_cls=10):
    if device is None:
        device = x_test.device
    n_batches = math.ceil(x_test.shape[0] / bs)
    logits = torch.zeros([x_test.shape[0], n_cls], device=device)
    l_logits = []
    
    with torch.no_grad():
        for counter in range(n_batches):
            x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
            output = model(x_curr)
            #l_logits.append(output.detach())
            logits[counter * bs:(counter + 1) * bs] += output.detach()
    
    return logits


def get_wc_acc(model, xs, y, bs=1000, device=None, eot_test=1, logger=None,
    loss=None):
    if device is None:
        device = x.device
    if logger is None:
        logger = Logger(None)
    if not loss is None:
        criterion_indiv = criterion_dict[loss]
    acc = torch.ones_like(y, device=device).float()
    x_adv = xs[0].clone()
    loss_best = -1. * float('inf') * torch.ones(y.shape[0], device=device)
    
    for x in xs:
        logits = get_logits(model, x, bs=bs, device=device).to(y.device)
        loss_curr = criterion_indiv(logits, y)
        pred_curr = logits.max(1)[1] == y
        ind = ~pred_curr * (loss_curr > loss_best) # misclassified points with higher loss
        x_adv[ind] = x[ind].clone()
        acc *= pred_curr
        ind = (acc == 1.) * (loss_curr > loss_best) # for robust points track highest loss
        x_adv[ind] = x[ind].clone()
        logger.log(f'[rob acc] cum={acc.mean():.1%} curr={pred_curr.float().mean():.1%}')
            
    return acc.mean(), x_adv


def eval_fast(model, x_test, y_test, norm='Linf', eps=8. / 255., savedir='./',
    bs=1000, short_version=False, log_path=None, eot_iter=1):
    #log_path = '{}/log_runs.txt'.format(savedir)
    
    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
        log_path=log_path
        )
    if short_version:
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    #adversary.apgd_targeted.n_target_classes = 3
    #adversary.apgd_targeted.n_iter = 20
    adversary.apgd.verbose = True
    adversary.apgd_targeted.verbose = True
    #adversary.square.verbose = True
    adversary.apgd.eot_iter = eot_iter
    adversary.apgd_targeted.eot_iter = eot_iter
    
    with torch.no_grad():
        x_adv = adversary.run_standard_evaluation(x_test, y_test, bs)

    other_utils.check_imgs(x_adv.to(x_test.device), x_test, norm)

    return x_adv

