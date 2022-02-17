import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Tuple
from functools import partial
import time

#from autopgd_base import L1_projection
from other_utils import L1_norm, L2_norm, L0_norm, Logger
from checks import check_zero_gradients


class GradientEstimator():
    def __init__(self, loss_fn, prior=None, n_iters=10, eta=1e-2,
        parall=False, type_est='finite_diff', n_fts=None, mu=.5):
        self.loss_fn = loss_fn
        self.prior = prior
        #self.model = model
        self.n_iters = n_iters
        self.eta = eta
        self.parallelize = parall
        self.type_est = type_est
        self.n_fts = n_fts
        self.mu = mu
    
    def estimate(self, x, loss_old=None):
        eta = self.eta
        with torch.no_grad():
            #if self.prior is None
            if self.type_est == 'finite_diff':
                prior = torch.zeros_like(x)
                
                if not self.parallelize:
                    for _ in range(self.n_iters):
                        s = torch.randn_like(x)
                        l1 = self.loss_fn(x + s * eta)
                        l2 = self.loss_fn(x - s * eta)
                        #print(l1)
                        prior += ((l1 - l2).view(-1, *[1]*(len(x.shape) - 1)) / (2 * eta) * s)
                else:
                    s = torch.randn([self.n_iters, *x.shape[1:]], device=x.device)
                    #print(s.shape)
                    z = x.repeat([2 * self.n_iters, 1, 1, 1])
                    z[:self.n_iters] += (s * eta)
                    z[self.n_iters:] -= (s * eta)
                    l = self.loss_fn(z)
                    l1, l2 = l[:self.n_iters], l[self.n_iters:]
                    prior += ((l1 - l2).view(-1, *[1]*(len(x.shape) - 1)) * s).sum(dim=0, keepdim=True) / (2 * eta)
                
                if self.prior is None:
                    return prior
                else:
                    mu = self.mu #[.75, .5][-1]
                    self.prior = self.prior * mu + prior * (1. - mu)
                    return self.prior.clone()

            elif self.type_est == 'unit':
                if self.parallelize:
                    s = torch.randn([self.n_iters, *x.shape[1:]], device=x.device)
                    s /= (L2_norm(s, keepdim=True) + 1e-12)
                    z = x.repeat([self.n_iters, *[1] * (len(x.shape) - 1)]) + s * eta
                    l = self.loss_fn(z)
                    grad = (l - loss_old).view(-1, *[1]*(len(x.shape) - 1)) * s / eta
                    return self.n_fts * grad.mean(0, keepdim=True)


def dlr_loss(x, y, reduction='none', i=None):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)


def cw_loss(x, y, i=None):
    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
        
    return -(x[torch.arange(x.shape[0]), y] - x_sorted[:, -2] * ind - \
        x_sorted[:, -1] * (1. - ind))


def gama_loss(x, y, x_orig, lmbd, i=1):
    return cw_loss(x, y) + lmbd(i) * L2_norm(x - x_orig)


def singlelogit_loss(x, y, i=None, targeted=False):
    u = torch.arange(x.shape[0])
    if not targeted:
        return -x[u, y]


def dlr_loss_targeted(x, y, y_target, i=None):
    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    return -(x[u, y] - x[u, y_target]) / (x_sorted[:, -1] - .5 * (
        x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)


criterion_dict = {'ce': lambda x, y, i=None: F.cross_entropy(x, y, reduction='none'),
    'dlr': dlr_loss,
    'cw': cw_loss,
    'gama': gama_loss,
    'maxlogit': singlelogit_loss,
    'dlr-targeted': dlr_loss_targeted
    }


def check_oscillation(x, j, k, y5, k3=0.75):
    t = torch.zeros(x.shape[1]).to(x.device)
    for counter5 in range(k):
      t += (x[j - counter5] > x[j - counter5 - 1]).float()

    return (t <= k * k3 * torch.ones_like(t)).float()


def orth_projection(a, b):
    u = b / (L2_norm(b, keepdim=True) + 1e-12)
    innerprod = (a * b).view(a.shape[0], -1).sum(-1).view(-1, *[1]*(len(a.shape) - 1))
    return innerprod * u


def apgd_twomodels(clf, ebm, x, y, norm='Linf', eps=8. / 255., n_iter=100,
    use_rs=False, loss='ce', verbose=True, type_grad='std', use_apgd=True,
    step_size=None):
    #assert not model.training
    device = x.device
    ndims = len(x.shape) - 1
    
    if not use_rs:
        x_adv = x.clone()
    else:
        raise NotImplemented
        if norm == 'Linf':
            t = torch.rand_like(x)
    
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    
    # set loss
    criterion_indiv = criterion_dict[loss]
    #
    lmbd = 1.
    
    # set parameters
    n_fts = math.prod(x.shape[1:])
    if norm in ['Linf', 'L2']:
        n_iter_2 = max(int(0.22 * n_iter), 1)
        n_iter_min = max(int(0.06 * n_iter), 1)
        size_decr = max(int(0.03 * n_iter), 1)
        k = n_iter_2 + 0
        thr_decr = .75
        alpha = 2.

    if use_apgd:
        step_size = alpha * eps * torch.ones([x.shape[0], *[1] * ndims],
            device=device)
    else:
        step_size = eps * step_size
    counter3 = 0
    
    #grad = torch.zeros_like(x)
    if type_grad == 'std':
        x_adv.requires_grad_()
        out_clf, out_ebm = clf(x_adv), ebm(x_adv)
        pred = out_clf.detach().max(1)[1] == y
        loss_clf = criterion_indiv(out_clf, y)
        loss_ebm = -1 * out_ebm
        loss_indiv = loss_clf + lmbd * loss_ebm #criterion_indiv(logits, y)
        loss = loss_indiv.sum()
        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        x_adv.detach_()
        loss_indiv.detach_()
        loss.detach_()
        loss_adv = loss_ebm.detach()
    elif type_grad == 'altern':
        x_adv.requires_grad_()
        out_clf, out_ebm = clf(x_adv), ebm(x_adv)
        pred = out_clf.detach().max(1)[1] == y
        loss_clf = criterion_indiv(out_clf, y)
        loss_ebm = -1 * out_ebm + 0.
        loss_indiv = loss_clf * 1. + loss_ebm * (~pred).float() #criterion_indiv(logits, y)
        loss = loss_indiv.sum()
        grad = torch.autograd.grad(loss, [x_adv])[0].detach()
        grad_best = grad.clone()
        x_adv.detach_()
        loss_indiv.detach_()
        #loss = loss_ebm.sum()
        loss.detach_()
    elif type_grad == 'orthogonal':
        x_adv.requires_grad_()
        out_clf, out_ebm = clf(x_adv), ebm(x_adv).squeeze()
        pred = out_clf.detach().max(1)[1] == y
        #print(out_clf.shape, out_ebm.shape)
        loss_clf = criterion_indiv(out_clf, y)
        loss_ebm = -out_ebm
        grad_clf = torch.autograd.grad(loss_clf.sum(), [x_adv])[0].detach()
        grad_ebm = torch.autograd.grad(loss_ebm.sum(), [x_adv])[0].detach()
        grad_clf_proj = grad_clf - orth_projection(grad_clf, grad_ebm)
        grad_ebm_proj = grad_ebm - orth_projection(grad_ebm, grad_clf)
        grad = grad_clf_proj * pred.float().view(-1, *[1] * ndims) + \
            grad_ebm_proj * (~pred).float().view(-1, *[1] * ndims)
        #print((grad - grad_ebm_proj).abs().sum(dim=(1, 2, 3)))
        x_adv.detach_()
        loss_indiv = loss_clf.detach()
        loss = loss_indiv.sum()
        loss_adv = loss_ebm.detach()
    
    grad_best = grad.clone()
    acc = out_clf.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_ebm.detach().clone() #loss_indiv.detach().clone()
    loss_best_last_check = loss_best.clone()
    reduced_last_check = torch.ones_like(loss_best)
    n_reduced = 0
    
    u = torch.arange(x.shape[0], device=device)
    x_adv_old = x_adv.clone().detach()
    
    for i in range(n_iter):
        ### gradient step
        if True: #with torch.no_grad()
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            loss_curr = loss.detach().mean()
            
            #a = 0.75 if i > 0 else 1.0
            a = 1.
            
            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                    x - eps), x + eps), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(
                    x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                    x - eps), x + eps), 0.0, 1.0)

            x_adv = x_adv_1 + 0.

        #print(f'it {i} [loss] clf={loss_clf.sum():.5f} ebm={loss_ebm.sum():.5f} [acc] curr={pred.float().mean():.1%}')
        
        ### get gradient
        if type_grad == 'std':
            x_adv.requires_grad_()
            out_clf, out_ebm = clf(x_adv), ebm(x_adv)
            loss_clf = criterion_indiv(out_clf, y)
            loss_ebm = out_ebm * -1.
            loss_indiv = loss_clf + lmbd * loss_ebm
            loss = loss_indiv.sum()
            if i < n_iter - 1:
                # save one backward pass
                grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            x_adv.detach_()
            loss_indiv.detach_()
            loss.detach_()
        elif type_grad == 'altern':
            x_adv.requires_grad_()
            out_clf, out_ebm = clf(x_adv), ebm(x_adv)
            pred = out_clf.detach().max(1)[1] == y
            loss_clf = criterion_indiv(out_clf, y)
            loss_ebm = -1 * out_ebm
            loss_indiv = loss_clf * 1. + loss_ebm * (~pred).float() #criterion_indiv(logits, y)
            loss = loss_indiv.sum()
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            #grad_best = grad.clone()
            x_adv.detach_()
            loss_indiv.detach_()
            #loss = loss_ebm.sum()
            loss.detach_()
        elif type_grad == 'orthogonal':
            x_adv.requires_grad_()
            out_clf, out_ebm = clf(x_adv), ebm(x_adv).squeeze()
            pred = out_clf.detach().max(1)[1] == y
            loss_clf = criterion_indiv(out_clf, y)
            loss_ebm = -out_ebm
            grad_clf = torch.autograd.grad(loss_clf.sum(), [x_adv])[0].detach()
            grad_ebm = torch.autograd.grad(loss_ebm.sum(), [x_adv])[0].detach()
            grad_clf_proj = grad_clf - orth_projection(grad_clf, grad_ebm)
            grad_ebm_proj = grad_ebm - orth_projection(grad_ebm, grad_clf)
            grad = grad_clf_proj * pred.float().view(-1, *[1] * ndims) + \
                grad_ebm_proj * (~pred).float().view(-1, *[1] * ndims)
            #print((grad - grad_ebm_proj).abs().sum(dim=(1, 2, 3)))
            x_adv.detach_()
            loss_indiv = loss_clf.detach()
            loss = loss_indiv.sum()
        
        pred = out_clf.detach().max(1)[1] == y
        acc_old = acc.clone()
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        #ind_pred = (pred == 0).nonzero().squeeze()
        #x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        ind_pred = (pred == 0) * (acc_old == 1.) + (~pred) * (
            loss_ebm.detach() > loss_adv) #.nonzero().squeeze()
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        loss_adv[ind_pred] = loss_ebm.detach().clone()[ind_pred]
        #
        if verbose:
            if use_apgd:
                str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                    step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                    step_size.mean())
            str_stats = f' - en {loss_ebm.detach().mean():.5f} - curr acc {pred.float().mean():.1%}'
            print('iteration: {} - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                i, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          if type_grad in ['std', 'orthogonal']:
              y1 = loss_indiv.detach().clone()
          elif type_grad in ['altern']:
              y1 = loss_ebm.detach().clone()
          loss_steps[i] = y1 + 0
          ind = (y1 > loss_best) * (~pred)
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k and use_apgd:
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

        #if acc.sum() == 0. and early_stop:
        #    break
    
    return x_best, acc, loss_best, x_best_adv





def get_init(init, model, x, y, n_cls=10, eta=None, norm=None, eps=None):
    if init == 'odi':
        w = torch.rand([x.shape[0], n_cls]).to(x.device) * 2. - 1.
        #w[torch.arange(x.shape[0]), y] = -1.
        L = lambda out1, w: (w * out1).sum(dim=1) / L2_norm(w * out1, keepdim=True)
        n_fts = math.prod(x.shape[1:])
        #x_orig = x.clone()
        #x_orig.requires_grad = True
        x_adv = torch.zeros_like(x)
        if isinstance(model, Tuple):
            clf, pfy_fn = model
            x_orig, _ = pfy_fn(x_adv)
            eot_iter = x_orig.shape[0] // x_adv.shape[0]
            x_orig.requires_grad = True
            loss = L(clf(x_orig), w.repeat([eot_iter, 1])).mean()
        else:
            x_orig = x.clone()
            x_orig.requires_grad = True
            #with torch.enable_grad()
            loss = L(model(2 * x_orig - 1), w).mean()
        grad_orig = torch.autograd.grad(loss, [x_orig])[0].detach()
        grad_curr = grad_orig.clone() #torch.zeros_like(grad_orig)
        grad_curr = grad_curr.view(-1, *x_adv.shape).mean(0)
        print('[odi] it 0, loss: {:.5f}'.format(loss.detach()))
        for it in range(1, 30): #10
            #x_adv = x_adv.detach() + eta * (grad_curr - grad_orig) / (L1_norm(
            #    grad_curr - grad_orig, keepdim=True) + 1e-10)
            if norm == 'Linf':
                x_adv = x_adv.detach() + eta * grad_curr.sign()
                x_adv = x + (x_adv - x).clamp(-eps, eps)
                x_adv.clamp_(0., 1.)
            elif norm == 'L2':
                x_adv = x_adv.detach() + eta * grad_curr / (L2_norm(grad_curr, keepdim=True) + 1e-12)
                currnorm = L2_norm(x_adv - x, keepdim=True)
                x_adv = x + (x_adv - x) / currnorm * torch.min(currnorm, eps * torch.ones_like(currnorm))
                x_adv.clamp_(0., 1.)
            elif norm == 'L1':
                grad_topk = grad_curr.abs().view(x.shape[0], -1).topk(
                    k=max(math.ceil(.1 * n_fts), 1), dim=-1)[0][:, -1].view(x.shape[0],
                    *[1]*(len(x.shape) - 1))
                grad_curr = grad_curr * (grad_curr.abs() > grad_topk).float()
                x_adv = x_adv.detach() + eta * grad_curr / (L1_norm(
                    grad_curr, keepdim=True) + 1e-10)
                x_adv += L1_projection(x, x_adv - x, eps)
            #
            '''x_adv.requires_grad = True
            with torch.enable_grad():
                loss = L(model(x_adv), w).mean()'''
            if isinstance(model, Tuple):
                clf, pfy_fn = model
                x_orig, _ = pfy_fn(x_adv)
                x_orig.requires_grad = True
                loss = L(clf(x_orig), w.repeat([eot_iter, 1])).mean()
                grad_curr = torch.autograd.grad(loss, [x_orig])[0].detach()
                grad_curr = grad_curr.view(-1, *x_adv.shape).mean(0)
            else:
                #x_orig = x.clone()
                x_adv.requires_grad = True
                #with torch.enable_grad()
                loss = L(model(2 * x_adv - 1), w).mean()
                grad_curr = torch.autograd.grad(loss, [x_adv])[0].detach()
            print('[odi] it {}, loss: {:.5f}'.format(it, loss.detach()))
        return x_adv.detach()


#def line_search(model, x, y, x_orig, n_steps=10)


def apgd_train(model, x, y, norm='Linf', eps=8. / 255., n_iter=100,
    use_rs=False, loss='ce', verbose=False, is_train=False, use_interm=False,
    use_ls=False, btls_pts=[], x_init=None, full_def=None, ge_mu=.5,
    use_btls=False, collect_steps=False, use_ge=False, ge_iters=10, ge_eta=1,
    ge_prior=None, logger=None, early_stop=True, pfy_fn=None, clf=None,
    mask=None, use_batch=False, stepsize=None, eot_iter=0, y_target=None):
    if isinstance(model, nn.Module):
        assert not model.training
    device = x.device
    ndims = len(x.shape) - 1
    n_cls = 10
    loss_name = loss
    
    if logger is None:
        logger = Logger()
    #
    
    if use_ls:
        print('using line search')
    
    if collect_steps:
        steps = {'imgs': [], 'loss': []}
        img = torch.ones(x.shape[0], device=x.device, dtype=bool)
    
    if mask is None:
        mask = torch.ones(x.shape[0], device=x.device).float()
    mask = mask.view(-1, *[1] * ndims)
    
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
    
    if not x_init is None:
        if isinstance(x_init, torch.Tensor):
            x_adv = x_init.clone()
            #print(eps)
            x_adv = x + (x_adv - x).clamp(-eps - 1e-5, eps + 1e-5)
        else:
            x_adv = get_init(x_init, [model, (clf, pfy_fn)][1], x, y, eta=eps / 4, norm=norm,
                eps=eps)
    
    x_adv = x_adv.clamp(0., 1.) * mask + x * (1. - mask)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([n_iter, x.shape[0]], device=device)
    loss_best_steps = torch.zeros([n_iter + 1, x.shape[0]], device=device)
    acc_steps = torch.zeros_like(loss_best_steps)
    loss_adv = -float('inf') * torch.ones(x.shape[0], device=device)
    
    # set loss
    if not loss in ['dlr-targeted', 'gama']:
        criterion_indiv = criterion_dict[loss]
    elif loss == 'gama':
        with torch.no_grad():
            output_orig = model(x)
        lmbd_gama = lambda x: max(1. * (n_iter - x * 2) / n_iter, 0)
        criterion_indiv = partial(gama_loss, x_orig=output_orig, lmbd=lmbd_gama)
    elif loss in ['dlr-targeted']:
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
        alpha = .1 if use_ge else 2. #.25 #2. #1e-5 #2e-4 #.01 #2.
        alpha = stepsize if not stepsize is None else alpha
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

    if use_ge:
        ge_iters_init = [2000, ge_iters][-1]
        if not x.shape[0] == 1:
            loss_fn = lambda x: criterion_indiv(model(x), y)
        else:
            #loss_fn = lambda x: criterion_indiv(model(x), y.repeat(2 * ge_iters_init))
            def loss_fn(x, bs=500):
                l_loss = []
                for counter in range(math.ceil(x.shape[0] / bs)):
                    x_curr = x[counter * bs:(counter + 1) * bs]
                    output_curr = criterion_indiv(model(x_curr), y.repeat(x_curr.shape[0]))
                    l_loss.append(output_curr)
                return torch.cat(l_loss, dim=0)
        
        #ge_iters_init = [5000, ge_iters][0]
        GradEst = GradientEstimator(loss_fn, prior=ge_prior, #torch.zeros_like(x)
            n_iters=ge_iters_init, eta=ge_eta, parall=x.shape[0] == 1, mu=ge_mu)
        #
        #raise NotImplemented
    
    #x_adv.requires_grad_()
    #grad = torch.zeros_like(x)
    #for _ in range(self.eot_iter)
    #with torch.enable_grad()
    startt = time.time()
    if not use_interm:
        if not use_ge:
            x_adv.requires_grad_()
            logits = model(x_adv)
            print(f'initial acc={(logits.max(1)[1] == y).float().mean():.1%}')
            loss_indiv = criterion_indiv(logits, y, i=0)
            loss = loss_indiv.sum()
            #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
            grad = torch.autograd.grad(loss, [x_adv])[0].detach()
            for i_eot in range(eot_iter):
                x_noisy = x_adv + torch.randn_like(x_adv) * ge_eta
                loss_noisy = criterion_indiv(model(x_noisy), y, i=0).sum()
                grad += torch.autograd.grad(loss_noisy, [x_noisy])[0].detach()
            #grad /= float(self.eot_iter)
            grad_best = grad.clone()
            x_adv.detach_()
            loss_indiv.detach_()
            loss.detach_()
        else:
            grad = GradEst.estimate(x_adv)
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
            GradEst.n_iters = ge_iters
            ge_iters_init = ge_iters
            print(f'setting ge iterations to {ge_iters} (first one was {ge_iters_init})')
    else:
        _, interm_x = pfy_fn(x_adv)
        if not use_ge:
            grad = torch.zeros_like(interm_x[0])
            #grad_ebm = grad.clone()
            #logits, interm_x = model(x_adv)
            loss_indiv = torch.zeros([interm_x[0].shape[0]], device=device)
            for x_step in interm_x:
                x_step.requires_grad_(True)
                logits = clf(x_step)
                loss_step = criterion_indiv(logits, y.repeat([x_step.shape[0] // x_adv.shape[0]]))
                #loss = loss_indiv.sum()
                grad_clf = torch.autograd.grad(loss_step.sum(), [x_step])[0].detach()
                grad += grad_clf #torch.autograd.grad(loss, [x_step])[0].detach()
                #x_step.detach_()
                loss_indiv = loss_step.detach()
                #loss.detach_()
                '''if not ebm is None:
                    lmbd = 1e1
                    en_indiv = ebm(2 * x_step - 1)
                    grad_ebm = torch.autograd.grad(en_indiv.sum(), [x_step])[0].detach()
                    grad += (grad_ebm * lmbd) #torch.autograd.grad(lmbd * en_indiv.sum(), [x_step])[0].detach()
                    #en_indiv.detach_()
                '''
                #print(grad_clf.abs().max(), grad_ebm.abs().max())
            assert not x_adv.requires_grad
            grad /= len(interm_x)
            print(grad.shape)
            
            loss = loss_indiv.detach().sum()
            #grad = grad.view(-1, *x_adv.shape).mean(0)
            print(logits.shape, loss_indiv.shape, grad.shape)
            #if not ebm is None:
            #    en = en_indiv.detach().view(-1, x_adv.shape[0]).mean(0)
        else:
            grad = GradEst.estimate(x_adv)
            logits = model(x_adv)
            loss_indiv = criterion_indiv(logits, y)
            loss = loss_indiv.sum()
            print(logits.shape, loss_indiv.shape, grad.shape)
            GradEst.n_iters = ge_iters
    grad *= mask
    
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
            
            if collect_steps:
                steps['imgs'].append(x_adv[img].cpu())
                steps['loss'].append(loss_indiv[img].cpu())
            
            a = 0.75 if i > 0 else 1.0
            #a = .5 if i > 0 else 1.
            a = 1. if use_ge else a
            
            if norm == 'Linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad) #+ 1e-2 * torch.randn_like(grad) #grad / (torch.sign(grad) + 1e-12)  #torch.sign(grad) #(grad.abs().view(grad.shape[0], -1).max(dim=-1
                    #)[0].view(-1, 1, 1, 1)
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
                
            if use_ls:
                loss_ls_best = -float('inf') * torch.ones(x.shape[0], device=x.device)
                stepsize_best = torch.zeros_like(loss_ls_best)
                x_adv_1 = x_adv.clone()
                stepsizes_ls = [3e-3 * c for c in range(1, 11)]
                for i_ls in stepsizes_ls: #range(3, 10 #[.5] + list(range(1, 100, 10))
                    x_curr = x_adv + eps * i_ls * grad.sign() #eps / 10 ** (-i_ls) * grad.sign()
                    x_curr = x + torch.min(torch.max(x_curr - x, -eps * e), eps * e)
                    x_curr.clamp_(0., 1.)
                    #x_pfy, _ = pfy_fn(x_curr)
                    logits_ls = model(x_curr)
                    loss_ls = criterion_indiv(logits_ls, y)
                    #loss_ls = loss_ls.view(-1, x.shape[0]).mean(0)
                    ind_ls = loss_ls > loss_ls_best
                    x_adv_1[ind_ls] = x_curr[ind_ls].clone()
                    loss_ls_best[ind_ls] = loss_ls[ind_ls].clone()
                    stepsize_best[ind_ls] = i_ls
                #
            
            elif use_btls:
                #print(grad.max(), grad.min())
                not_improved = torch.ones_like(y, dtype=bool)
                eta = eps * 4
                x_adv_1 = x_adv.clone()
                while not_improved.any():
                    eta /= 5
                    x_curr = x_adv + eta * grad.sign() #+ 1e-1 * torch.randn_like(grad)
                    x_curr = x + torch.min(torch.max(x_curr - x, -eps * e), eps * e)
                    x_curr.clamp_(0., 1.)
                    x_pfy, _ = pfy_fn(x_curr)
                    logits_ls = model(x_pfy)
                    loss_ls = criterion_indiv(logits_ls, y.repeat([
                        x_step.shape[0] // x_adv.shape[0]]))
                    loss_ls = loss_ls.view(-1, x.shape[0]).mean(0)
                    #ind_ls = loss_ls > loss_ls_best
                    x_adv_1[not_improved] = x_curr[not_improved].clone()
                    #loss_ls_best[ind_ls] = loss_ls[ind_ls].clone()
                    not_improved = loss_ls < loss_indiv
                    if eta < 1e-10:
                        break
            
            
            x_adv = x_adv_1 + 0.

        ### get gradient
        if not use_interm:
            if not use_ge:
                x_adv.requires_grad_()
                #grad = torch.zeros_like(x)
                #for _ in range(self.eot_iter)
                #with torch.enable_grad()
                logits = model(x_adv)
                loss_indiv = criterion_indiv(logits, y, i=i)
                loss = loss_indiv.sum()

                #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                if i < n_iter - 1:
                    # save one backward pass
                    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
                    for i_eot in range(eot_iter):
                        x_noisy = x_adv + torch.randn_like(x_adv) * ge_eta
                        loss_noisy = criterion_indiv(model(x_noisy), y, i=0).sum()
                        grad += torch.autograd.grad(loss_noisy, [x_noisy])[0].detach()
                #grad /= float(self.eot_iter)
                x_adv.detach_()
                loss_indiv.detach_()
                loss.detach_()
            else:
                grad = GradEst.estimate(x_adv)
                logits = model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
        else:
            #logits, interm_x = model(x_adv)
            if not use_ge:
                _, interm_x = pfy_fn(x_adv)
                #interm_x = [x_adv.clone()] + interm_x
                grad = torch.zeros_like(interm_x[0])
                loss_indiv = torch.zeros([interm_x[0].shape[0]], device=device)
                #is_miscl = ~torch.zeros_like(loss_indiv, dtype=bool)
                for x_step in interm_x:
                    x_step.requires_grad_(True)
                    logits = clf(x_step) #model(x_step)
                    #pred = logits.detach().max(1)[1] == y.repeat(eot_iter)
                    #is_miscl *= pred
                    loss_step = criterion_indiv(logits, y)
                    #loss = loss_indiv.sum()
                    grad_clf = torch.autograd.grad(loss_step.sum(), [x_step])[0].detach()
                    grad += grad_clf #* (~is_miscl).float().view(-1, *[1] * ndims) #torch.autograd.grad(loss, [x_step])[0].detach()
                    #x_step.detach_()
                    #loss_indiv.detach_()
                    #loss.detach_()
                    loss_indiv = loss_step.detach()
                    '''if not ebm is None:
                        en_indiv = ebm(2 * x_step - 1)
                        grad_ebm = torch.autograd.grad(en_indiv.sum(), [x_step])[0].detach()
                        grad += (grad_ebm * lmbd) #(~acc).float().view(-1, 1, 1, 1)
                        #grad += torch.autograd.grad(en_indiv.sum(), [x_step])[0].detach()
                        #en_indiv.detach_()
                    #print(grad_clf.abs().max(), grad_ebm.abs().max())
                    '''
                assert not x_adv.requires_grad
                grad /= len(interm_x)
                #logits = logits.view(-1, x_adv.shape[0], logits.shape[-1]).mean(0)
                #loss_indiv = loss_indiv.view(-1, x_adv.shape[0]).mean(0)
                loss = loss_indiv.detach().sum()
                #grad = grad.view(-1, *x_adv.shape).mean(0)
                check_zero_gradients(grad)
                zeros_in_grad = (grad == 0.).view(grad.shape[0], -1).sum(-1)
                #print(zeros_in_grad)
                #if not ebm is None:
                #    en = en_indiv.detach().view(-1, x_adv.shape[0]).mean(0)
            else:
                grad = GradEst.estimate(x_adv)
                logits = model(x_adv)
                loss_indiv = criterion_indiv(logits, y)
                loss = loss_indiv.sum()
        grad *= mask
        
        pred = logits.detach().max(1)[1] == y
        acc_old = acc.clone()
        if not use_batch:
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            ind_pred = (pred == 0).nonzero().squeeze()
            #x_best_adv[ind_pred] = x_adv[ind_pred]
            ind = (loss_indiv.detach() > loss_best).nonzero().squeeze()
        else:
            ind_pred = (pred.float().mean() < acc_old.float().mean()) + (
                (pred.float().mean() < acc_old.float().mean()) * (loss > loss_best.sum())) #(pred == 0) * (acc_old == 1.) + (~pred) * (loss_indiv.detach().clone() > loss_adv) #.nonzero().squeeze()
            if ind_pred:
                acc = pred.clone()
            ind_pred = ind_pred.repeat(x.shape[0])
            ind = ind_pred.clone()
            #
        x_best_adv[ind_pred] = x_adv[ind_pred] + 0.
        loss_adv[ind_pred] = loss_indiv.detach().clone()[ind_pred]
        if verbose:
            str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
                step_size.mean(), topk.mean() * n_fts) if norm in ['L1'] else ' - step size: {:.5f}'.format(
                step_size.mean() if not use_ls else stepsize_best.mean())
            #str_stats = f' - en {en.mean():.5f}' if not ebm is None else str_stats
            if mask.sum() < x.shape[0]:
                str_stats += f' - mask acc: {(acc * mask[:, 0, 0, 0]).sum():.0f} loss: {(loss_indiv * mask.squeeze()).sum():.3f}'
            if loss_name in ['maxlogit']:
                str_stats += f' - margin: {cw_loss(logits, y).mean():.3f}'
            currt = time.time() - startt
            logger.log('iteration: {} ({:.1f} s) - best loss: {:.6f} curr loss {:.6f} - robust accuracy: {:.2%}{}'.format(
                i, currt, loss_best.sum(), loss_curr, acc.float().mean(), str_stats))
            #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
        
        ### check step size
        if True: #with torch.no_grad()
          y1 = loss_indiv.detach().clone()
          loss_steps[i] = y1 + 0
          #ind = (y1 > loss_best).nonzero().squeeze()
          #ind = ind_pred.clone()
          x_best[ind] = x_adv[ind].clone()
          grad_best[ind] = grad[ind].clone()
          loss_best[ind] = y1[ind] + 0
          loss_best_steps[i + 1] = loss_best + 0

          counter3 += 1

          if counter3 == k and not (use_ls or use_btls) and (alpha == 2. or alpha > .5):
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

        if (acc * mask.squeeze()).sum() == 0. and early_stop:
            break
    
    if collect_steps:
        torch.save(steps, f'./results/rn-18/collect_steps_21.pth')
    
    return x_best, acc, loss_best, x_best_adv


def apgd_interm_restarts(model, x, y, norm='Linf', eps=8. / 255., n_iter=10,
    loss='ce', verbose=False, use_interm=True, n_restarts=1, #ebm=None,
    x_init=None, use_ge=False, ge_iters=10, ge_eta=1, ge_prior=False,
    log_path=None, early_stop=False, pfy_fn=None, clf=None,
    use_mask=False, use_batch=False, stepsize=None, use_ls=False, ge_mu=.5,
    eot_iter=0):
    acc = torch.ones([x.shape[0]], dtype=bool, device=x.device
        ) if not isinstance(x_init, torch.Tensor) else \
        model(x_init).max(1)[1] == y
    print(f'initial accuracy={acc.float().mean():.1%}')
    x_adv = x.clone()
    x_best = x.clone()
    loss_best = -float('inf') * torch.ones_like(acc).float()
    #print(loss_best.shape)
    if loss in ['dlr-targeted']:
        with torch.no_grad():
            output = model(x)
        outputsorted = output.sort(-1)[1]
    else:
        y_target = None
    for i in range(n_restarts):
        if acc.sum() > 0:
            if isinstance(x_init, torch.Tensor):
                x_init_curr = x_init[acc]
                if len(x_init.shape) == 3:
                    x_init.unsqueeze_(0)
            else:
                x_init_curr = x_init
            if i > 10:
                x_init_curr = 'odi'
            if loss in ['dlr-targeted']:
                y_target = outputsorted[:, -(i + 2)]
                y_target = y_target[acc]
                print(f'target class {i + 2}')
            if not use_mask:
                x_best_curr, _, loss_curr, x_adv_curr = apgd_train(model, x[acc], y[acc],
                    use_interm=use_interm, n_iter=min(n_iter * 2 ** 0 if i > -1  else 5, 4000),
                    use_rs=True, stepsize=stepsize, eot_iter=eot_iter,
                    verbose=verbose, loss=loss, use_ls=use_ls, #ebm=ebm,
                    x_init=x_init_curr, #x_init[acc] if i < 2 else None
                    #full_def=full_def,
                    pfy_fn=pfy_fn, clf=clf,
                    eps=eps, norm=norm, use_batch=use_batch,
                    use_btls=False * (i > 0), collect_steps=False, #i == n_restarts - 1
                    use_ge=use_ge, ge_iters=ge_iters, ge_eta=ge_eta, ge_mu=ge_mu,
                    ge_prior=torch.zeros_like(x) if ge_prior else None,
                    logger=Logger(log_path), early_stop=early_stop,
                    y_target=y_target)
            else:
                x_mask = x.clone() #if x_init is None else x_init.clone()
                if not x_init is None:
                    x_mask[~acc] = x_init[~acc] + 0.
                for c in range(x.shape[0]):
                    if not acc[c]:
                        continue
                    print(c)
                    mask = torch.zeros([x.shape[0]], device=x.device).float()
                    mask[c] = 1.
                    x_best_curr, _, loss_curr, x_mask = apgd_train(model, x.clone(), y,
                        use_interm=use_interm, n_iter=n_iter * 2 ** i if i > -1  else 5, use_rs=True,
                        verbose=verbose, loss=loss, use_ls=use_ls, #ebm=ebm,
                        x_init=x_mask, #x_init[acc] if i < 2 else None
                        #full_def=full_def,
                        pfy_fn=pfy_fn, clf=clf, mask=mask,
                        eps=eps, norm=norm,
                        use_btls=False * (i > 0), collect_steps=False, #i == n_restarts - 1
                        use_ge=use_ge, ge_iters=ge_iters, ge_eta=ge_eta,
                        ge_prior=torch.zeros_like(x) if ge_prior else None,
                        logger=Logger(log_path), early_stop=early_stop,
                        )
                    #acc_curr = model(x_adv_curr).max(1)[1] == y[acc]
                x_adv_curr = x_mask.clone()
            acc_curr = model(x_adv_curr).max(1)[1] == y[acc]
            succs = torch.nonzero(acc).squeeze()
            if len(succs.shape) == 0:
                succs.unsqueeze_(0)
            #succs = succs[~acc_curr]
            #print(succs)
            x_adv[succs[~acc_curr]] = x_adv_curr[~acc_curr].clone()
            ind = succs[acc_curr * (loss_curr > loss_best[acc])]
            x_best[ind] = x_best_curr[acc_curr * (loss_curr > loss_best[acc])].clone()
            loss_best[ind] = loss_curr[acc_curr * (loss_curr > loss_best[acc])].clone()
            try:
                acc[acc] = acc_curr.clone()
            except:
                ind_acc = torch.nonzero(acc).squeeze()
                acc[ind_acc] = acc_curr.clone()
            '''print(loss_best[acc].shape)
            print(loss_curr.shape)
            print(ind, loss_best[ind], loss_curr[acc_curr * (loss_curr > loss_best[acc])])
            loss_best[ind] = loss_curr[acc_curr * (loss_curr > loss_best[acc])].clone()'''
            print(f'restart {i + 1} robust accuracy={acc.float().mean():.1%}')
            #print(loss_best[acc].min())
            print(torch.nonzero(acc).squeeze())
    x_best[~acc] = x_adv[~acc].clone()
    
    
    return x_adv, x_best



