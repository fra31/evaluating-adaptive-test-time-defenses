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
import other_utils
from other_utils import L2_norm, Linf_norm
import robustbench as rb
#from aid_purifier.code_final.adaptive_opt import criterion_dict


def load_dataset(dataset: str, n_ex: int = 1000, device: str = 'cuda',
    data_dir: str = None #'/home/scratch/datasets/CIFAR10'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    #device = 'cuda:0'
    if data_dir is None:
        data_dir = f'/home/scratch/datasets/{dataset.upper()}'
        
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
    
    return logits #torch.cat(l_logits, dim=0)


def eval_maxloss(model, x_test, y_test, norm='Linf', eps=8. / 255., savedir='./',
    bs=1000, loss='ce', x_init=None, loss_fn=None, eot_iter=1, n_iter=100,
    n_restarts=1, verbose=False, seed=None, log_path=None):
    if log_path is None:
        log_path = '{}/log_runs_maxloss.txt'.format(savedir)

    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
        log_path=log_path, seed=seed)
    #adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    #adversary.apgd.n_target_classes = 3
    adversary.apgd.n_iter = n_iter #200
    adversary.apgd.n_restarts = n_restarts #1
    adversary.apgd.loss = loss
    adversary.apgd.eot_iter = eot_iter
    adversary.apgd.verbose = verbose
    adversary.apgd.seed = seed
    
    x_adv = torch.zeros_like(x_test)
    
    with torch.no_grad():
        #x_adv = adversary.run_standard_evaluation(x_test, y_test, bs)
        for counter in range(math.ceil(x_test.shape[0] / bs)):
            x_curr = x_test[counter * bs:(counter + 1) * bs].clone()
            y_curr = y_test[counter * bs:(counter + 1) * bs].clone()
            x_adv_curr = adversary.apgd.perturb(x_curr, y_curr,
                best_loss=True, x_init=x_init, loss_fn=loss_fn)
            x_adv[counter * bs:(counter + 1) * bs] += x_adv_curr.to(x_adv.device)
    
    acc = rb.utils.clean_accuracy(model, x_adv.cuda(), y_test.cuda(),
        batch_size=bs)
    print('robust accuracy={:.1%}'.format(acc))
    
    other_utils.check_imgs(x_adv.to(x_test.device), x_test, norm)

    return x_adv


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


def orth_projection(a, b):
    u = b / (other_utils.L2_norm(b, keepdim=True) + 1e-12)
    innerprod = (a * b).view(a.shape[0], -1).sum(-1).view(-1, *[1]*(len(a.shape) - 1))
    return innerprod * u


def get_2D_losslandscape(model, xs, ys, delta=8. / 255., n_pts=100,
    dir1s=None, dir2s=None, normal=None, loss='ce', seed=None):
    assert len(xs.shape) == 4
    intv = torch.linspace(-delta, delta, n_pts, device=xs.device)
    vals = torch.empty([xs.shape[0], n_pts, n_pts], device=xs.device)
    loss_fn = criterion_dict[loss] #lambda x, y: F.cross_entropy(x, y, reduction='none')
    if not seed is None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        print(torch.rand(1).item())
    for i, x in enumerate(xs):
        x.unsqueeze_(0)
        assert len(x.shape) == 4
        if dir1s is None:
            dir1 = torch.randn_like(x)
        else:
            dir1 = dir1s[i].unsqueeze(0)
        if dir2s is None:
            dir2 = torch.randn_like(x)
            dir2 = dir2 - orth_projection(dir2, dir1)
        if normal == 'Linf':
            dir1 /= Linf_norm(dir1, keepdim=True)
            dir2 /= Linf_norm(dir2, keepdim=True)
        elif normal == 'L2':
          dir1 /= L2_norm(dir1, keepdim=True)
          dir2 /= L2_norm(dir2, keepdim=True)
        #vals = torch.empty([n_pts, n_pts], dtype=float)
        #intv = torch.linspace(-delta, delta, n_pts, device=x.device)
        for c in range(n_pts):
            x_curr = x.repeat([n_pts, 1, 1, 1]) + dir1 * intv[c].item() + \
                dir2 * intv.view(-1, 1, 1, 1) 
            with torch.no_grad():
                output = model(x_curr)
            vals[i][c] = loss_fn(output, ys[i].repeat(n_pts))
        print(f'image {i} done')
    return intv.cpu(), vals.cpu()


def eval_with_square(model, x_test, y_test, norm='Linf', eps=8. / 255., savedir='./',
    bs=100, loss='ce', eot_iter=1, n_iter=10000, seed=None,
    opt_loss=True, n_restarts=1, eta=None, log_path=None):
    if log_path is None:
        log_path = '{}/log_runs.txt'.format(savedir)
    
    adversary = autoattack.AutoAttack(model, norm=norm, eps=eps,
        log_path=log_path, seed=seed)
    adversary.attacks_to_run = [#'apgd-ce',
        #'apgd-t'
        #'fab-t',
        'square'
        ]
    #adversary.apgd_targeted.n_target_classes = 3
    #adversary.apgd_targeted.n_iter = 20
    #adversary.apgd.n_iter = 25
    adversary.square.verbose = True
    adversary.square.loss = loss
    adversary.square.eot_iter = eot_iter
    adversary.square.return_all = True
    adversary.square.n_queries = n_iter
    adversary.square.n_restarts = n_restarts
    adversary.square.opt_loss = opt_loss
    adversary.square.rescale_schedule = False
    if not eta is None:
        adversary.square.add_noise = True
        adversary.square.noise_eta = eta
    
    with torch.no_grad():
        #x_adv = adversary.run_standard_evaluation(x_test, y_test, bs)
        x_adv = adversary.square.perturb(x_test, y_test)
    
    other_utils.check_imgs(x_adv.to(x_test.device), x_test, norm)

    return x_adv


def get_batch(x, y, bs, counter, device='cuda'):
    x_curr = x[counter * bs:(counter + 1) * bs].to(device)
    y_curr = y[counter * bs:(counter + 1) * bs].to(device)
    return x_curr, y_curr


def clean_acc_with_eot(model, x_test, y_test, bs, eot_test=1, method='logits',
    device='cuda', n_cls=10, return_acc_dets=False):
    with torch.no_grad():
        #startt = time.time()
        #output = get_logits(adp_fn, x_test, bs=100)
        output = torch.zeros([x_test.shape[0], n_cls], device=device)
        acc_dets = torch.zeros([x_test.shape[0]], device=device)
        for _ in range(eot_test):
            #output_curr = adp_fn(x_test)
            output_curr = get_logits(model, x_test, bs=bs, device=device)
            acc_dets += (output_curr.max(1)[1] == y_test.to(device)).float()
            if method == 'softmax':
                output += F.softmax(output_curr, 1)
            elif method == 'logits':
                output += output_curr.clone()
        #tott = time.time() - startt
        acc = output.max(1)[1] == y_test.to(device)
    if return_acc_dets:
        #print(acc_dets.long())
        return acc, acc_dets
    return acc


def clf_with_eot(x, model, eot_test, method='softmax', device='cuda',
    n_cls=10, bs=1000):
    output = torch.zeros([x.shape[0], n_cls], device=device)
    for _ in range(eot_test):
        #output_curr = adp_fn(x_test)
        output_curr = get_logits(model, x, bs=bs, device=device)
        if method == 'softmax':
            output += F.softmax(output_curr, 1)
        elif method == 'logits':
            output += output_curr.clone()
    return output / eot_test


def get_wc_doubleeot(model, xs, y, eot_def, eot_test, method='softmax',
    device='cuda', n_cls=10, bs=1000):
    acc = torch.ones_like(y, device=device).float()
    x_adv = xs[0].clone()
    for x in xs:
        pred_cum = torch.zeros_like(acc)
        for i in range(eot_test):
            output_curr = clf_with_eot(x, model, eot_def, method=method,
                device=device, n_cls=n_cls, bs=bs)
            pred_curr = output_curr.max(1)[1]
            pred_cum += pred_curr.to(device) == y.to(device)
            #print(f'eot iter={i + 1}')
        pred_cum /= eot_test
        ind = (pred_cum < acc).to(x.device)
        x_adv[ind] = x[ind] + 0.
        acc[ind] = pred_cum[ind].clone()
        print(f'[rob acc] cum={(acc > .5).float().mean():.1%} curr={(pred_cum > .5).float().mean():.1%}')
    acc = (acc > .5).float()
    return acc.mean(), x_adv


def join_batches(savedir, runname, indices=[0, 0, 0]):
    xs = []
    runnames = []
    for ind in range(*indices):
        runnames.append(f'{savedir}/{runname}'.replace('indices',
            f'{ind}-{min(ind + indices[-1], indices[1])}'))
        assert os.path.exists(runnames[-1]), f'{runnames[-1]} missing'
        print(runnames[-1])
    for c in runnames:
        x = torch.load(c)
        xs.append(x)
    x = torch.cat(xs, dim=0)
    newrunname = f'{savedir}/{runname}'.replace('indices', f'{indices[0]}-{indices[1]}')
    print(newrunname, x.shape[0])
    torch.save(x, newrunname)

