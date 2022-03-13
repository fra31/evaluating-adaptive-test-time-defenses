import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple
from functools import partial
from random import shuffle

try:
    from other_utils import L1_norm, L2_norm, L0_norm, Logger
except ImportError:
    from autoattack.other_utils import L1_norm, L2_norm, L0_norm, Logger


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


def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
      t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def get_target_features(model, x, y):
    """ for an input x returns the features of a point z which is classified
        in the second most likely class for x if x correctly classified, in
        the first one otherwise
    """
    logits, fts = model.forward(x, return_fts=True)
    pred = logits.sort(dim=-1)[1]
    fts_target = fts.clone()
    ind = [c for c in range(x.shape[0])]
    shuffle(ind)
    for c in range(x.shape[0]):
        if pred[c][-1] != y[c]:
            continue
        fts_secondclass = False
        for i in ind:
            if pred[i][-1] == pred[c][-2]:
                fts_target[c] = fts[i].clone()
                fts_secondclass = True
                break
        # make sure that fts from another class are used
        if not fts_secondclass:
            for i in ind:
                if pred[i][-1] != y[c]:
                    fts_target[c] = fts[i].clone()
                    break
    return fts_target


def apgd_train(model, x, y, norm='Linf', eps=8. / 255., n_iter=100,
    use_rs=False, loss='ce', verbose=False, is_train=False, logger=None,
    early_stop=True, y_target=None, fts_target=None):
    assert not model.training
    device = x.device
    ndims = len(x.shape) - 1
    n_cls = 10
    
    if logger is None:
        logger = Logger()
    loss_name = loss + ''
    
    # initialization
    if not use_rs:
        x_adv = x.clone()
    else:
        #raise NotImplemented
        if norm == 'Linf':
            t = (torch.rand_like(x) - .5) * 2. * eps
            x_adv = (x + t).clamp(0., 1.)
        elif norm == 'L2':
            t = torch.randn_like(x)
            x_adv = x + eps * t / (L2_norm(t, keepdim=True) + 1e-12)
            x_adv.clamp_(0., 1.)
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    loss_adv = -float('inf') * torch.ones(x.shape[0], device=device)
    
    # set loss
    if not loss in ['dlr-targeted']:
        criterion_indiv = criterion_dict[loss]
    else:
        assert not y_target is None
        criterion_indiv = partial(criterion_dict[loss], y_target=y_target)
    
    # set params
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.
    elif norm in ['L1']:
        k = max(int(.04 * n_iter), 1)
        init_topk = .05 if is_train else .2
        topk = init_topk * torch.ones([x.shape[0]], device=device)
        sp_old =  n_fts * torch.ones_like(topk)
        adasp_redstep = 1.5
        adasp_minstep = 10.
        alpha = 1.
    
    step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
        device=device)
    counter3 = 0

    x_adv.requires_grad_()
    if not loss in ['l2', 'l1', 'linf']:
        logits = model(x_adv)
        loss_indiv = criterion_indiv(logits, y)
        loss = loss_indiv.sum()
    else:
        logits, fts = model.forward(x_adv, return_fts=True)
        loss_indiv = criterion_indiv(fts, fts_target)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    x_adv.detach_()
    loss_indiv.detach_()
    loss.detach_()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0
    
    u = torch.arange(x.shape[0], device=device)
    e = torch.ones_like(x)
    x_adv_old = x_adv.clone().detach()
    
    for i in range(n_iter):
        ### gradient step
        if True: #with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach() #.mean()
            
            a = 0.75 if i > 0 else 1.0
            #a = .5 if i > 0 else 1.
            
            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            elif norm == 'L2':
                x_adv_1 = x_adv + step_size * grad / (L2_norm(grad,
                    keepdim=True) + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (L2_norm(x_adv_1 - x,
                    keepdim=True) + 1e-12) * torch.min(eps * torch.ones_like(x),
                    L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

            elif norm == 'L1':
                grad_topk = grad.abs().view(x.shape[0], -1).sort(-1)[0]
                topk_curr = torch.clamp((1. - topk) * n_fts, min=0, max=n_fts - 1).long()
                grad_topk = grad_topk[u, topk_curr].view(-1, *[1]*(len(x.shape) - 1))
                sparsegrad = grad * (grad.abs() >= grad_topk).float()
                x_adv_1 = x_adv + step_size * sparsegrad.sign() / (
                    sparsegrad.sign().abs().view(x.shape[0], -1).sum(dim=-1).view(
                    -1, 1, 1, 1) + 1e-10)
                
                delta_u = x_adv_1 - x
                delta_p = L1_projection(x, delta_u, eps)
                x_adv_1 = x + delta_u + delta_p
                
            elif norm == 'L0':
                L1normgrad = grad / (grad.abs().view(grad.shape[0], -1).sum(
                    dim=-1, keepdim=True) + 1e-12).view(grad.shape[0], *[1]*(
                    len(grad.shape) - 1))
                x_adv_1 = x_adv + step_size * L1normgrad * n_fts
                x_adv_1 = L0_projection(x_adv_1, x, eps)
                # TODO: add momentum
                
            
            x_adv = x_adv_1 + 0.

        ### get gradient
        x_adv.requires_grad_()
        if not loss_name in ['l2', 'l1', 'linf']:
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
        else:
            logits, fts = model.forward(x_adv, return_fts=True)
            loss_indiv = criterion_indiv(fts, fts_target)
            loss = loss_indiv.sum()
        if i < n_iter - 1:
            # save one backward pass
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        
        # collect points and stats
        pred = logits.detach().max(1)[1] == y
        acc_old = acc.clone()
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        ind_pred = (pred == 0) * (acc_old == 1.) + (~pred) * (
            loss_indiv.detach().clone() > loss_adv) #.nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        loss_adv[ind_pred] = loss_indiv.detach().clone()[ind_pred]
        
        # logging
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean())
            
            logger.log('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1 + 0
          ind = (y1 > loss_best).nonzero().squeeze()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k:
              if norm in ['Linf', 'L2']:
                  fl_oscillation = check_oscillation(loss_steps, i, k,
                      loss_best, k3=thr_decr)
                  fl_reduce_no_impr = (1. - reduced_last_check) * (
                      loss_best_last_check >= loss_best).float()
                  fl_oscillation = torch.max(fl_oscillation,
                      fl_reduce_no_impr)
                  reduced_last_check = fl_oscillation.clone()
                  loss_best_last_check = loss_best.clone()

                  if fl_oscillation.sum() > 0:
                      ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                      step_size[ind_fl_osc] /= 2.0
                      n_reduced = fl_oscillation.sum()

                      x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                      grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()
                  
                  counter3 = 0
                  k = max(k - size_decr, n_iter_min)
              
              elif norm == 'L1':
                  # adjust sparsity
                  sp_curr = L0_norm(x_best - x)
                  fl_redtopk = (sp_curr / sp_old) < .95
                  topk = sp_curr / n_fts / 1.5
                  step_size[fl_redtopk] = alpha * eps
                  step_size[~fl_redtopk] /= adasp_redstep
                  step_size.clamp_(alpha * eps / adasp_minstep, alpha * eps)
                  sp_old = sp_curr.clone()
              
                  x_adv[fl_redtopk] = x_best[fl_redtopk].clone()
                  grad[fl_redtopk] = grad_best[fl_redtopk].clone()
              
                  counter3 = 0

        if acc.sum() == 0. and early_stop:
            break
    
    return x_best, acc, loss_best, x_best_adv


def apgd_restarts(model, x, y, norm='Linf', eps=8. / 255., n_iter=10,
    loss='ce', verbose=False, n_restarts=1, log_path=None, early_stop=False):
    """ run apgd with the option of restarts
    """
    acc = torch.ones([x.shape[0]], dtype=bool, device=x.device) # run on all points
    x_adv = x.clone()
    x_best = x.clone()
    loss_best = -float('inf') * torch.ones_like(acc).float()
    y_target = None
    fts_target = None
    if loss in ['dlr-targeted']:
        with torch.no_grad():
            output = model(x)
        outputsorted = output.sort(-1)[1]
        n_target_classes = 4 # max number of target classes to use
    elif loss in ['l2', 'l1', 'linf']:
        with torch.no_grad():
            fts_target = get_target_features(model, x, y)
    
    for i in range(n_restarts):
        if acc.sum() > 0:
            if loss in ['dlr-targeted']:
                y_target = outputsorted[:, -(i % n_target_classes + 2)]
                y_target = y_target[acc]
                print(f'target class {i % n_target_classes + 2}')
            elif loss in ['l2', 'l1', 'linf']:
                with torch.no_grad():
                    fts_target = get_target_features(model, x[acc], y[acc])
            
            x_best_curr, _, loss_curr, x_adv_curr = apgd_train(model, x[acc], y[acc],
                n_iter=n_iter, use_rs=True, verbose=verbose, loss=loss,
                eps=eps, norm=norm, logger=Logger(log_path), early_stop=early_stop,
                y_target=y_target, fts_target=fts_target)
            
            acc_curr = model(x_adv_curr).max(1)[1] == y[acc]
            succs = torch.nonzero(acc).squeeze()
            if len(succs.shape) == 0:
                succs.unsqueeze_(0)
            x_adv[succs[~acc_curr]] = x_adv_curr[~acc_curr].clone()
            # old version
            '''ind = succs[acc_curr * (loss_curr > loss_best[acc])]
            x_best[ind] = x_best_curr[acc_curr * (loss_curr > loss_best[acc])].clone()
            loss_best[ind] = loss_curr[acc_curr * (loss_curr > loss_best[acc])].clone()'''
            # new version
            ind = succs[loss_curr > loss_best[acc]]
            x_best[ind] = x_best_curr[loss_curr > loss_best[acc]].clone()
            loss_best[ind] = loss_curr[loss_curr > loss_best[acc]].clone()
            
            ind_new = torch.nonzero(acc).squeeze()
            acc[ind_new] = acc_curr.clone()
            
            print(f'restart {i + 1} robust accuracy={acc.float().mean():.1%}')
            
    # old version
    #x_best[~acc] = x_adv[~acc].clone()
    
    
    return x_adv, x_best



