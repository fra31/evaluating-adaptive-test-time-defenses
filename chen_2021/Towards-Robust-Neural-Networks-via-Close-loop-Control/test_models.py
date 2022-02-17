import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from Adversarial_attack import fgsm, Random, pgd, CW_attack, Manifold_attack

def testing(test_loader, model, step_size, eps, attack='None', device=None):    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        if attack == 'None':
            images_ = inputs
        elif attack == 'fgsm':
            images_ = fgsm(inputs, labels, eps, criterion, model)
        elif attack == 'random':
            images_ = Random(inputs, labels, eps, criterion, model)
        elif attack == 'pgd':
            images_ = pgd(model, inputs, labels, criterion, num_steps=20, step_size=step_size, eps=eps)
        elif attack == 'cw':
            print('Processing CW attack on batch:', i)
            CW = CW_attack(model)
            images_ = CW.attack(inputs, labels, eps)
        outputs = model(images_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    return accuracy

def manifold_attack(test_loader, model, eps, basis, device):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    total_step = len(test_loader)
    test_loss = 0
    correct = 0
    total = 0
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        print('Processing CW attack on batch:', i)
        Man_attack = Manifold_attack(model, basis)
        images_ = Man_attack.attack(inputs, labels, eps)
        outputs = model(images_)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    accuracy = 100.*correct/total
    print('Testing accuracy:', accuracy)
    return accuracy        
    