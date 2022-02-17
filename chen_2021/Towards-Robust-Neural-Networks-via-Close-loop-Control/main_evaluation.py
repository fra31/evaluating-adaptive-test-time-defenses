import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import sys
import numpy as np
import random
import argparse

from model import *
from train_models import training
from test_models import testing, manifold_attack

from Linear_control_funs import Linear_Control
from NonLinear_control_funs import Nonlinear_Control

# For reproducibility
torch.manual_seed(999)
np.random.seed(999)
random.seed(999)
torch.cuda.manual_seed_all(999)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic=True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Towards robust neural networks via close-loop control')
parser.add_argument('--data_set', type=str, default='cifar10', help='Can be either cifar10 | cifar100')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--number_of_workers', default=0, type=int, help='number_of_workers')
parser.add_argument('--perturbation_type', type=str, default='None', 
                    choices=['None', 'fgsm', 'pgd', 'cw', 'manifold'],
                    help='Choose a perturbation type between [standard, fgsm, pgd, cw, manifold]')
parser.add_argument('--perturbation_eps', type=float, default=8.,
                    help='Choose a perturbation magnitude')
parser.add_argument('--defense_type', default='None', type=str,
                    choices=['None', 'layer_wise_projection', 'linear_pmp', 'non_linear_pmp'],
                    help='Choose a defense type between [None, layer_wise_projection, linear_pmp, non_linear_pmp]')
parser.add_argument('--pmp_lr', type=float, default=0.01,
                    help='Choose a learning rate for the PMP defense')
parser.add_argument('--pmp_maximum_iterations', type=int, default=10,
                    help='Choose a maximum iterations for the PMP defense')
parser.add_argument('--model_selection', default='resnet20_standard', type=str,
                    choices=['resnet20_standard', 'resnet20_fgsm', 'resnet20_label_smooth, resnet20_pgd'],
                    help='Choose a defense type between [resnet20_standard, resnet20_fgsm, resnet20_label_smooth, resnet20_pgd]')
parser.add_argument('--pmp_select_parameters', action="store_true",
                    help='Searching for the optimal lr and iterations for the pmp')

args = parser.parse_args()
workers = args.number_of_workers
data_set = args.data_set
batch_size = args.batch_size
pert_type = args.perturbation_type
pert_epsilon = args.perturbation_eps
defense_type = args.defense_type
pmp_lr = args.pmp_lr
pmp_max_iter = args.pmp_maximum_iterations
model_selection = args.model_selection
pmp_param_selection = args.pmp_select_parameters

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
if data_set == 'cifar10':
    num_classes = 10
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)
elif data_set == 'cifar100':
    num_classes = 100
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False,
        num_workers=workers, pin_memory=True)

# Select a trained model
print('==> Building model..')
net = resnet20(num_classes=num_classes)
net = net.to(device)
if model_selection == 'resnet20_standard':
    if data_set == 'cifar10':
        net.load_state_dict(torch.load('models_cifar10/resnet20_model.ckpt', map_location=device))
    elif data_set == 'cifar100':
        net.load_state_dict(torch.load('models_cifar100/resnet20_model.ckpt', map_location=device))
elif model_selection == 'resnet20_fgsm':
    if data_set == 'cifar10':
        net.load_state_dict(torch.load('models_cifar10/resnet20_fgsm.ckpt', map_location=device))
    elif data_set == 'cifar100':
        net.load_state_dict(torch.load('models_cifar100/resnet20_fgsm.ckpt', map_location=device))
elif model_selection == 'resnet20_label_smooth':
    if data_set == 'cifar10':
        net.load_state_dict(torch.load('models_cifar10/resnet20_label_smooth.ckpt', map_location=device))
    elif data_set == 'cifar100':
        net.load_state_dict(torch.load('models_cifar100/resnet20_label_smooth.ckpt', map_location=device))
elif model_selection == 'resnet20_pgd':
    if data_set == 'cifar10':
        net.load_state_dict(torch.load('models_cifar10/resnet20_pgd.ckpt', map_location=device))
    elif data_set == 'cifar100':
        net.load_state_dict(torch.load('models_cifar100/resnet20_pgd.ckpt', map_location=device))

# Select a hyper-parameters of the pmp
if pmp_param_selection:
    Lin = Linear_Control(net)
    Lin.compute_Princ_basis(train_loader)
    Lin.from_basis_projection()
    Lin.PMP_testing(test_loader)
    sys.exit("Searching pmp hyper-parameters done") 

# Testing against various perturbations without defense
if defense_type == 'None':
    testing(test_loader, net, step_size=pert_epsilon/4., eps=pert_epsilon, attack=pert_type, device=device)
    
# Testing against various perturbations with layer-wise-projection
elif defense_type == 'layer_wise_projection':
    Lin = Linear_Control(net)
    Lin.compute_Princ_basis(train_loader)
    Lin.from_basis_projection()
    Lin.testing(test_loader, eps=pert_epsilon, step_size=pert_epsilon/4., attack=pert_type, defense=defense_type)
    
# Testing against various perturbations with pmp consisted of linear embedding
elif defense_type == 'linear_pmp':
    Lin = Linear_Control(net)
    Lin.compute_Princ_basis(train_loader)
    Lin.from_basis_projection()
    Lin.testing(test_loader, eps=pert_epsilon, step_size=pert_epsilon/4., attack=pert_type, defense='pmp', 
                lr=pmp_lr, max_iter=pmp_max_iter)

# Testing against various perturbations with pmp consisted of nonlinear embedding
elif defense_type == 'non_linear_pmp':
    # Import the trained auto-encoders sequentially,
    # e.g. [Auto_encoder_0, Auto_encoder_1,...,Auto_encoder_4]
    embedding_funs = []
    Lin = Nonlinear_Control(net, embedding_funs)
    Lin.testing(test_loader, eps=pert_epsilon, step_size=pert_epsilon/4., attack=pert_type, defense='pmp', 
                lr=pmp_lr, max_iter=pmp_max_iter)




