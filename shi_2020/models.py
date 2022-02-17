import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from resnet import *
from wide_resnet import *

import os

def load_model(name, dataset, n_class=10, in_channel=3, save_dir=None, substitute=False):
    if name == 'fcnet':
        model = FCNet(n_class=n_class, in_dim=784, hidden_dim=128)
    elif name == 'cnet':
        model = CNet(n_class=n_class, in_channel=in_channel)
    elif name == 'ae':
        model = AutoEncoder(n_class=n_class, in_dim=784, hidden_dim=128)
    elif name == 'cae':
        model = ConvAutoEncoder(n_class=n_class, in_channel=in_channel)
    elif name == 'resnet':
        model = ResNet_(18, n_class)
    elif name == 'wide-resnet':
        model = Wide_ResNet_(28, 10, 0.3, n_class)
    elif name == 'resnet-rot':
        model = ResNet(n_class=n_class)
    elif name == 'wide-resnet-rot':
        model = WResNet(n_class=n_class)
    else:
        raise TypeError("Unrecognized model name: {}".format(name))

    if dataset == 'cifar10':
        model.add_normalizer(normalizer(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]))
    elif dataset == 'cifar100':
        model.add_normalizer(normalizer(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]))

    if save_dir is not None:
        if substitute:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'substitute_{}.pth'.format(name)), map_location='cpu'))
        else:
            model.load_state_dict(torch.load(os.path.join(save_dir, 'latest_model.pth'), map_location='cpu'))
    return model


class normalizer(nn.Module):

    def __init__(self, mean, std):
        super(normalizer, self).__init__()
        self.mean = torch.FloatTensor(mean)[:, None, None]
        self.std = torch.FloatTensor(std)[:, None, None]

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)


class add_noise(nn.Module):

    def __init__(self, std):
        super(add_noise, self).__init__()
        self.std = std

    def forward(self, x):
        return (x + torch.randn_like(x)*self.std).clamp(0,1)


class FCNet(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128, nonlinear='Relu'):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, return_reps=False):

        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.cls(x)
        return x


class CNet(nn.Module):
    def __init__(self, n_class, in_channel=3, hidden_dim=1024):

        super(CNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, return_reps=False):

        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 7*7*64)
        x = self.relu(self.fc1(x))
        x = self.cls(x)
        return x


class AutoEncoder(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128):

        super(AutoEncoder, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim*2)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc4 = nn.Linear(hidden_dim*2, in_dim)
        self.cls = nn.Linear(hidden_dim, n_class)
        
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.5).clamp(0,1)
        size = x.shape
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        if return_reps:
            return x

        l = self.cls(x)
        self.pred = l
        
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        self.r = x.reshape(*size)
        if self.aux:
            return self.r

        return l


class ConvAutoEncoder(nn.Module):

    def __init__(self, n_class, in_channel=3, hidden_dim=1024, out_channel=None):

        super(ConvAutoEncoder, self).__init__()
        if not out_channel:
            out_channel = in_channel
        self.conv1 = nn.Conv2d(in_channel, 32, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.fc1 = nn.Linear(7*7*64, hidden_dim)
        self.cls = nn.Linear(hidden_dim, n_class)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.fc2 = nn.Linear(hidden_dim, 7*7*64)
        self.conv3 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(32, out_channel, 3, stride=2, padding=1, output_padding=1)

        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        size = x.shape
        if add_noise:
            x = (x + torch.randn_like(x)*0.5).clamp(0,1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))

        x = x.view(-1, 7*7*64)
        x = self.relu(self.fc1(x))

        if return_reps:
            return x

        l = self.cls(x)
        self.pred = l
        
        x = self.relu(self.fc2(x))
        x = x.view(-1, 64, 7, 7)
        x = self.relu(self.conv3(x))
        x = self.sigmoid(self.conv4(x))
        
        self.r = x.reshape(*size)
        if self.aux:
            return self.r
        return l


class FCNet_rotate(nn.Module):

    def __init__(self, n_class, in_dim, hidden_dim=128):

        super(FCNet_rotate, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32)
        self.fc3 = nn.Linear(32, n_class)
        self.fc4 = nn.Linear(32, 4)
        self.relu = nn.ReLU(inplace=True)
        self.aux = False

    def forward(self, x):

        size = x.shape
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        self.pred_deg = self.fc4(x)
        if self.aux:
            return self.pred_deg
        return self.fc3(x)


class ResNet(nn.Module):

    def __init__(self, n_class):

        super(ResNet, self).__init__()
        self.resnet = ResNet_(18, n_class)
        self.fc1 = nn.Linear(64, 4)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.1).clamp(0,1)
        l = self.resnet(x)

        if return_reps:
            return self.resnet.x
        self.pred_deg = self.fc1(self.resnet.x)
        if self.aux:
            return self.pred_deg
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.resnet.add_normalizer(normalizer) 


class WResNet(nn.Module):

    def __init__(self, n_class, k=10):

        super(WResNet, self).__init__()
        self.resnet = Wide_ResNet_(28, k, 0.3, n_class)
        self.fc1 = nn.Linear(k*64, 4)
        self.aux = False

    def forward(self, x, add_noise=False, return_reps=False):

        if add_noise:
            x = (x + torch.randn_like(x)*0.1).clamp(0,1)
        # normalization in wide-resnet
        l = self.resnet(x)

        if return_reps:
            return self.resnet.x
            
        self.pred_deg = self.fc1(self.resnet.x)
        if self.aux:
            return self.pred_deg
        self.pred = l
        return l

    def add_normalizer(self, normalizer):
        self.resnet.add_normalizer(normalizer) 
