import torch
import torch.nn.functional as F
#import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from torch.utils.data import Dataset
from typing import Callable, Dict, Optional, Sequence, Set, Tuple
import math
import os

import autoattack
try:
    import other_utils
    from other_utils import L2_norm, Linf_norm
except:
    from autoattack import other_utils
    from autoattack.other_utils import L2_norm, Linf_norm
import robustbench as rb


def load_dataset(dataset: str, n_ex: int = 1000, device: str = 'cuda',
    data_dir: str = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    if dataset in ['cifar10', 'cifar100']:
        dataset_: rb.model_zoo.enums.BenchmarkDataset = rb.model_zoo.enums.BenchmarkDataset(dataset)
        #threat_model_: rb.model_zoo.enums.ThreatModel = rb.model_zoo.enums.ThreatModel(args.threat_model)
        prepr = transforms.Compose([transforms.ToTensor()])
        
        x_test, y_test = rb.data.load_clean_dataset(dataset_, n_ex, data_dir,
            prepr=prepr)
            
    elif dataset == 'svhn':
        dataset = datasets.SVHN(root=data_dir,
                                split='test', #train=False
                                transform=transforms.Compose([transforms.ToTensor()]),
                                download=True)
        x_test, y_test = rb.data._load_dataset(dataset, n_ex)
    return x_test.to(device), y_test.to(device)


def get_logits(model, x_test, bs=1000, device=None, track_grad=False,
    n_cls=10):
    if device is None:
        device = x_test.device
    n_batches = math.ceil(x_test.shape[0] / bs)
    logits = torch.zeros([x_test.shape[0], n_cls], device=device)
    l_logits = []
    
    if not track_grad:
        with torch.no_grad():
            for counter in range(n_batches):
                x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
                output = model(x_curr)
                #l_logits.append(output.detach())
                logits[counter * bs:(counter + 1) * bs] += output.detach()
                #print(f'batch={counter + 1}')
    else:
        for counter in range(n_batches):
            #output = model(x_test[counter * bs:(counter + 1) * bs])
            x_curr = x_test[counter * bs:(counter + 1) * bs].to(device)
            output = model(x_curr)
            #l_logits.append(output)
            logits[counter * bs:(counter + 1) * bs] += output
    
    return logits


def get_wc_acc(model, xs, y, bs=1000, device=None, eot_test=1):
    if device is None:
        device = x.device
    acc = torch.ones_like(y, device=device).float()
    x_adv = xs[-1].clone()
    if eot_test == 1:
        for x in xs:
            pred_curr = get_logits(model, x, bs=bs, device=device)
            pred_curr = pred_curr.max(1)[1]
            pred_curr = pred_curr.to(device) == y
            ind = (acc == 1.) * ~pred_curr
            x_adv[ind] = x[ind].clone()
            acc *= pred_curr
            print(f'[rob acc] cum={acc.mean():.1%} curr={pred_curr.float().mean():.1%}')
    else:
        for x in xs:
            pred_cum = torch.zeros_like(acc)
            for i in range(eot_test):
                pred_curr = get_logits(model, x, bs=bs, device=device).max(1)[1]
                pred_cum += pred_curr.to(device) == y.to(device)
                #print(f'eot iter={i + 1}')
            pred_cum /= eot_test
            ind = (pred_cum < acc).to(x.device)
            x_adv[ind] = x[ind] + 0.
            acc[ind] = pred_cum[ind].clone()
            print(f'[rob acc] cum={(acc > .5).float().mean():.1%} curr={(pred_cum > .5).float().mean():.1%}')
        acc = (acc > .5).float()
    
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


def dlr_loss(x, y, reduction='none'):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def cw_loss(x, y):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind))


def dlr_loss_targeted(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


criterion_dict = {'ce': lambda x, y: F.cross_entropy(x, y, reduction='none'),
    'dlr': dlr_loss,
    'cw': cw_loss,
    'dlr-targeted': dlr_loss_targeted,
    'l2': lambda x, y: -1. * L2_norm(x - y) ** 2.,
    'l1': lambda x, y: -1. * L1_norm(x - y),
    'linf': lambda x, y: -1. * (x - y).abs().max(-1)[0],
    }


def get_batch(x, y, bs, counter, device='cuda'):
    x_curr = x[counter * bs:(counter + 1) * bs].to(device)
    y_curr = y[counter * bs:(counter + 1) * bs].to(device)
    return x_curr, y_curr


def clean_acc_with_eot(model, x_test, y_test, bs, eot_test=1, method='logits',
    device='cuda', n_cls=10):
    """ it aggregates the output (logits or softmax) of multiple runs
    """
    with torch.no_grad():
        output = torch.zeros([x_test.shape[0], n_cls], device=device)
        for _ in range(eot_test):
            output_curr = get_logits(model, x_test, bs=bs, device=device)
            
            if method == 'softmax':
                output += F.softmax(output_curr, 1)
            elif method == 'logits':
                output += output_curr.clone()
        
        acc = output.max(1)[1] == y_test.to(device)
    
    return acc

