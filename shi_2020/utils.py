import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import grad, Variable

from attacks import *
# from defenses import *
from criterions import *

import os
import copy
import pickle
import numpy as np
from collections import deque


def train(model, train_loader, criterion, optimizer, scheduler, device):
    '''
    scheduler not used
    '''
    model.train()
    error, acc = 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = criterion(pred, y)
        error += loss.item()

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(train_loader)
    acc = acc / len(train_loader.dataset)
    print('train loss: {} / acc: {}'.format(error, acc))

def train_with_auxiliary(model, train_loader, joint_criterion, optimizer, scheduler, device):
    
    model.train()
    error, acc, error_aux, acc_aux = 0., 0., 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        joint_loss, (loss, aux_loss) = joint_criterion(model, X=X, y=y)
        error += loss.item()
        error_aux += aux_loss.item()
        
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()
            
        acc += (model.pred[:y.shape[0]].max(dim=1)[1] == y).sum().item()
        if joint_criterion.keywords['aux_criterion'].__name__ == 'rotate_criterion':
            acc_aux += (model.pred_deg.max(dim=1)[1].cpu() == torch.arange(4)[:, None].repeat(1, X.shape[0]).flatten()).sum().item()

    error = error / len(train_loader)
    error_aux = error_aux / len(train_loader)
    acc = acc / len(train_loader.dataset)
    if joint_criterion.keywords['aux_criterion'].__name__ == 'rotate_criterion':
        acc_aux = acc_aux / len(train_loader.dataset) / 4
        print('train loss: {} / acc: {} / err-aux: {} / acc-aux: {}'.format(error, acc, error_aux, acc_aux))
    else:
        print('train loss: {} / acc: {} / err-aux: {}'.format(error, acc, error_aux))

def train_adversarial(model, train_loader, criterion, attack, optimizer, device):

    model.train()

    error, acc = 0., 0.
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        if attack is not None:
            model.eval()
            delta = attack(model, criterion, X, y)
            model.train()
            pred = model(X+delta)
        else:
            pred = model(X)

        loss = criterion(pred, y)
        error += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc += (pred.max(dim=1)[1] == y).sum().item()
    error = error / len(train_loader)
    acc = acc / len(train_loader.dataset)
    print('train loss: {} / acc: {}'.format(error, acc))

def evaluate(model, eval_loader, criterion, device):

    model.eval()

    error, acc = 0., 0.
    with torch.no_grad():
        for X, y in eval_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)

            loss = criterion(pred, y)
            error += loss.item()

            acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(eval_loader)
    acc = acc / len(eval_loader.dataset)
    print('val loss: {} / acc: {}'.format(error, acc))

    return acc

def evaluate_auxiliary(model, eval_loader, aux_criterion, device):

    model.eval()
    error, acc = 0., 0.
    with torch.no_grad():
        for X, y in eval_loader:
            X, y = X.to(device), y.to(device)
            loss = aux_criterion(model, X)
            error += loss.item()
    error = error / len(eval_loader)
    print('val loss: {}'.format(error))

def evaluate_adversarial(model, loader, criterion, aux_criterion, attack, purify, device):

    model.eval()
    error, acc = 0., 0.
    clean, adv, df = 0., 0., 0.
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        if 'model' in attack.keywords.keys(): # if substitute model is specified
            delta = attack(criterion=criterion, X=X, y=y)
        else:
            delta = attack(model, criterion, X, y)

        X_pfy = purify(model, aux_criterion, X+delta)
        pred = model(X_pfy)

        loss = nn.functional.cross_entropy(pred, y)
        error += loss.item() 
        acc += (pred.max(dim=1)[1] == y).sum().item()

    error = error / len(loader)
    acc = acc / len(loader.dataset)
    print('adv loss: {} / acc: {}'.format(error, acc))

    return acc

def save_reps(model, train_loader, criterion, attack, defense, save_dir, device):

    if os.path.exists(os.path.join(save_dir, 'reps.pkl')):
        print('representation already exists!')
        return
    model.eval()
    reps, labels = [], []
    
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        if attack is None:
            reps.append(model(X, return_reps=True))
        else:
            delta = attack(model, criterion, X, y)
            if defense is not None:
                inv_delta = defense(model, X=X+delta)
                reps.append(model((X+delta+inv_delta).clamp(0,1), return_reps=True))
            else:
                reps.append(model(X+delta, return_reps=True))
        labels.append(y)
    reps = torch.cat(reps, dim=0).detach()
    labels = torch.cat(labels, dim=0)
    if save_dir is None:
        return reps, labels
    with open(os.path.join(save_dir, 'reps_pfy.pkl'), 'wb') as f:
        pickle.dump(reps.cpu().numpy(), f)

def save_logits(model, train_loader, save_dir, device):

    if os.path.exists(os.path.join(save_dir, 'logits.pkl')):
        return
    model.eval()
    logits = []
    with torch.no_grad():
        for X, _ in train_loader:
            X = X.to(device).flatten(start_dim=1)
            logits.append(model(X))
    logits = torch.cat(logits, dim=0).cpu().numpy()
    with open(os.path.join(save_dir, 'logits.pkl'), 'w') as f:
        pickle.dump(logits, f)

def save_file(buffer, file_dir):
    buffer = torch.cat(buffer, dim=0).cpu().numpy()
    with open(file_dir, 'wb') as f:
        pickle.dump(buffer, f)

def jacobian_augment(model, train_loader, lmbda, device):

    model.eval()
    new_data = []
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        X.requires_grad_()
        l = model(X)
        loss = l[torch.arange(y.shape[0]), y].sum()
        loss.backward()
        new_data.append(X + lmbda * X.grad.sign())
    return torch.cat(new_data, dim=0).detach().cpu()
