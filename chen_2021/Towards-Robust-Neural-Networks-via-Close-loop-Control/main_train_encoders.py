import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
import argparse

from model import *
from auto_encoder import ConvAutoencoder, training

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Train auto-encoders')
parser.add_argument('--data_set', type=str, default='cifar10', help='Can be either cifar10 | cifar100')
parser.add_argument('--train_index', type=int, default=0, help='Which auto-encoder to train')
parser.add_argument('--epochs', type=int, default=300, help='number of epochs to train')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='l2 regularization')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
parser.add_argument('--batch_size', default=128, type=int, help='batch_size')
parser.add_argument('--learning_rate', default=0.01, type=float, help='learning_rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--number_of_workers', default=0, type=int, help='number_of_workers')
parser.add_argument('--model_selection', default='resnet20_standard', type=str,
                    choices=['resnet20_standard', 'resnet20_fgsm', 'resnet20_label_smooth, resnet20_pgd'],
                    help='Choose a defense type between [resnet20_standard, resnet20_fgsm, resnet20_label_smooth, resnet20_pgd]')

args = parser.parse_args()
encoder_index = args.train_index
workers = args.number_of_workers
data_set = args.data_set
epochs = args.epochs
start_epoch = args.start_epoch
batch_size = args.batch_size
learning_rate = args.learning_rate
momentum = args.momentum
weight_decay = args.weight_decay
model_selection = args.model_selection


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


# Pre-define channel dimensions
# The channel dimensions are given based on the used ResNet structure
# For different net structure, other channel dimensions can be specificed
if encoder_index == 0:
    channels = [3, 18, 36]
elif encoder_index == 1:
    channels = [16, 36, 72]
elif encoder_index == 2:
    channels = [16, 36, 72]
elif encoder_index == 3:
    channels = [32, 72, 156]
elif encoder_index == 4:
    channels = [64, 128, 256]    

# Initializing the auto-encoder
Auto_encoder = ConvAutoencoder(channels)

# Train the auto-encoder based on given encoder_index
training(train_loader, test_loader, Auto_encoder, net, epochs, encoder_index, learning_rate, device)

    
    
    
    
    
    
    
    
    
    
    