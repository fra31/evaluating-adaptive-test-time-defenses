import torch
import torch.nn as nn

from torchdiffeq import odeint_adjoint as odeint

from wideresnet import WideResNet_save


endtime = 5
fc_dim = 64
act = torch.sin 
f_coeffi = -1
layernum = 0
tol = 1e-3


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


def create_model(device):
    model = WideResNet_save(fc_dim, 34, 10, widen_factor=10, dropRate=0.0)
    odefunc = ODEfunc_mlp(0)
    feature_layers = [ODEBlock(odefunc)] 
    fc_layers = [MLP_OUT_Linear()]
    model = nn.Sequential(model, *feature_layers, *fc_layers).to(device)
    return model

