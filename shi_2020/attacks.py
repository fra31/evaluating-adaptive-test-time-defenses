import torch
import torch.nn as nn
import torch.optim as optim

from math import exp
from functools import partial

from cw_attack import L2Adversary
from df_attack import DeepFool

def empty(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    return torch.zeros_like(X)

def inject_noise(X, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    return (X + torch.randn_like(X) * epsilon).clamp(*bound) - X

def fgsm(model, criterion, X, y=None, epsilon=0.1, bound=(0,1)):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    if y is None:
        loss = criterion(model, X + delta)
    else:
        try:
            criterion_name = criterion.func.__name__
            if criterion_name == 'second_order':
                loss = criterion(model, X + delta, y)
            else:
                loss = criterion(model(X + delta), y)
        except:
            loss = criterion(model(X + delta), y)
    loss.backward()
    if y is None:
        delta = epsilon * delta.grad.detach().sign()
    else:
        delta = epsilon * delta.grad.detach().sign()
    return (X + delta).clamp(*bound) - X

def pgd_linf(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, randomize=False,
    return_interm=False):
    """ Construct PGD adversarial examples on the examples X"""
    if randomize:
        delta = torch.rand_like(X, requires_grad=True)
        delta.data = delta.data * 2 * epsilon - epsilon
    else:
        delta = torch.zeros_like(X, requires_grad=True)

    int_delta = []
    
    for t in range(num_iter):
        if y is None:
            loss = criterion(model, X + delta)
        else:
            try:
                criterion_name = criterion.func.__name__
                if criterion_name == 'second_order':
                    loss = criterion(model, X + delta, y)
                else:
                    loss = criterion(model(X + delta), y)
            except:
                loss = criterion(model(X + delta), y)
        loss.backward()
        delta.data = (delta + step_size*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        delta.grad.zero_()
        int_delta.append(delta.clone().detach())
    
    delta.data = (X + delta).clamp(*bound) - X
    return delta.detach(), torch.stack(int_delta, dim=-1)

def bpda(model, criterion, X, y=None, epsilon=0.1, bound=(0,1), step_size=0.01, num_iter=40, purify=None):

    delta = torch.zeros_like(X)
    for t in range(num_iter):

        X_pfy = purify(model, X=X + delta).detach()
        X_pfy.requires_grad_()

        loss = criterion(model(X_pfy), y)
        loss.backward()

        delta.data = (delta + step_size*X_pfy.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.data = (X + delta).clamp(*bound) - X
        X_pfy.grad.zero_()

    return delta.detach()

def cw(model, criterion, X, y=None, epsilon=0.1, num_classes=10):
    delta = L2Adversary()(model, X.clone().detach(), y, num_classes=num_classes).to(X.device) - X
    delta_norm = torch.norm(delta, p=2, dim=(1,2,3), keepdim=True) + 1e-4
    delta_proj = (delta_norm > epsilon) * delta / delta_norm * epsilon + (delta_norm < epsilon) * delta
    return delta_proj

def df(model, criterion, X, y=None, epsilon=0.1):
    delta = DeepFool()(model, X.clone().detach()).clamp(0,1) - X
    delta_norm = torch.norm(delta, p=2, dim=(1,2,3), keepdim=True)
    delta_proj = (delta_norm > epsilon) * delta / delta_norm * epsilon + (delta_norm < epsilon) * delta
    return delta_proj
