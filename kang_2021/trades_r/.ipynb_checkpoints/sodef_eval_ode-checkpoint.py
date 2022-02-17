import argparse
import copy
import logging
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from preactresnet import PreActResNet18
from wideresnet import WideResNet,WideResNet1,WideResNet_save
from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)
from autoattack import AutoAttack
import torch.utils.data as data
device = torch.device('cuda:0') 
from torchdiffeq import odeint_adjoint as odeint

endtime = 5

fc_dim = 64
act = torch.sin 
f_coeffi = -1
layernum = 0
tol = 1e-3


saved_temp = torch.load('./EXP/nips_model/full.pth')
statedic_temp = saved_temp['state_dict']




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x)[0].cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)





class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


    
class ODEfunc_mlp(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 256)
        self.act1 = act
        self.fc2 = ConcatFC(256, 256)
        self.act2 = act
        self.fc3 = ConcatFC(256, 64)
        self.act3 = act
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = f_coeffi*self.fc1(t, x)
        out = self.act1(out)
        out = f_coeffi*self.fc2(t, out)
        out = self.act2(out)
        out = f_coeffi*self.fc3(t, out)
        out = self.act3(out)
        
        return out

    
    
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, endtime]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=tol, atol=tol)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)

    
class MLP_OUT_Linear(nn.Module):

    def __init__(self):
        super(MLP_OUT_Linear, self).__init__()
        self.fc0 = nn.Linear(fc_dim, 10)
    def forward(self, input_):
#         h1 = F.relu(self.fc0(input_))
        h1 = self.fc0(input_)
        return h1


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)

        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)



def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


        

args = get_args()
nepochs = 100
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


train_loader, test_loader, train_loader__, test_dataset = get_loaders(args.data_dir, args.batch_size)


args = get_args()
nepochs = 100
batches_per_epoch = 128

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)




model = WideResNet_save(fc_dim, 34, 10, widen_factor=10, dropRate=0.0)


odefunc = ODEfunc_mlp(0)
feature_layers = [ODEBlock(odefunc)] 
fc_layers = [MLP_OUT_Linear()]


model = nn.Sequential(model, *feature_layers, *fc_layers).to(device)


model.load_state_dict(statedic_temp)

print(model)
model.to(device)
model.eval()


l = [x for (x, y) in test_loader]
x_test = torch.cat(l, 0)
l = [y for (x, y) in test_loader]
y_test = torch.cat(l, 0)


    
iii = 0
x_test = x_test[1024*iii:1024*(iii+1),...]
y_test = y_test[1024*iii:1024*(iii+1),...]
    
# x_test = x_test[256*iii:256*(iii+1),...]
# y_test = y_test[256*iii:256*(iii+1),...]   



# print('run_standard_evaluation_individual', 'Linf')
# print('run_standard_evaluation', 'Linf')
print('run_standard_evaluation', 'L2')


# epsilon = 8 / 255.
# adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')

epsilon = 0.5
adversary = AutoAttack(model, norm='L2', eps=epsilon, version='standard')



# adversary.attacks_to_run = ['fab']
# self.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
# adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
#     adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'square']
# adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
# adversary.attacks_to_run = ['apgd-t']
#     adversary.attacks_to_run = ['fab-t','square']
# adversary.attacks_to_run = ['fab-t']
# adversary.attacks_to_run = ['square']



X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)
# X_adv = adversary.run_standard_evaluation_individual(x_test, y_test, bs=256)

