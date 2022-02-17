import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import defaultdict

from Adversarial_attack import fgsm, Random, pgd, CW_attack, Manifold_attack

from tabulate import tabulate

PCA_INDEX = [0, 1, 4, 7, 10]

class Vectorize(nn.Module):
    def __init__(self):
        super(Vectorize, self).__init__()
        return
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
def get_seq_model(model): 
    # Convert nn.Module into a sequential model
    model.eval()
    seq = nn.Sequential(
            nn.Sequential(model.conv1,model.bn1,torch.nn.ReLU()),
            *list(list(model.layer1.children())+
                     list(model.layer2.children())+
                     list(model.layer3.children())),
            nn.Sequential(torch.nn.AvgPool2d(kernel_size=8),Vectorize(),model.linear))
    return seq

def SVD(A, thd):
    # Singularization value decomposition
    # Return truncated right basis
    # thd takes [0., 1.]
    U, Sigma, V = torch.svd(A)
    sum_singu = Sigma.sum()
    singus = 0.
    count = 0
    for sigma in Sigma:
        singus += sigma
        count += 1
        if (singus / sum_singu) > thd:
            break
    print('The rank is:', count)
    return V[:, 0:count], Sigma[0:count], count

def rSVD(X, r, q=1, p=5):
    # Randomized singularization value decomposition
    ny = X.shape[1]
    P = torch.randn(ny, r + p)
    Z = X @ P
    for k in range(q):
        Z = X @ (X.t() @ Z)
    
    Q, R = torch.qr(Z, some=True)
    
    Y = Q.t() @ X
    UY, S, VT = torch.svd(Y)
    U = Q @ UY
    return U, S, VT

def PCA_Projection(data, projection):
    # Projecting data onto the range of given basis
    data_shape = data.shape
    data = data.view(data_shape[0], -1)
    mean = data.mean(dim=0, keepdim=True)
    data = data - mean
    data_proj = torch.mm(data, projection)
    data_proj = data_proj + mean
    return data_proj.view(data_shape) 

# The main control functions
# For given neural network and embedding functions, //
# we recomend to run <PMP_testing>, which uses the first batch to search for //
# the learning rate and maximum iterations for the PMP algorithm. //
    
# The same hyper-parameters (learning rate and maximum iterations) must be used
# against all perturbations
    
class Linear_Control:
    def __init__(self, model, thd=0.99, max_num_samples=6500, pca_index=PCA_INDEX,
                 c_regularization=0.):
        model.eval()
        self.model = model
        self.seq_model = get_seq_model(self.model)
        self.THD = thd
        self.MAX_NUM_SAMPLES = max_num_samples
        self.PCA_INDEX = pca_index
        self.C_REGULARIZATION = c_regularization
        self.Princ_vec = defaultdict(list)
        self.Proj = defaultdict(list)

        # The control parameters that were used in the last prediction
        self.last_Conts = []
    
    # Performing principle component analysis on input and hidden layers
    def compute_Princ_basis(self, data_loader, device=None):
        with torch.no_grad():
            datai = torch.zeros(0).to(device)
            for idx, (inputs, labels) in enumerate(data_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                # Samples are selected if those can result in correct predictions
                outputs = self.model(inputs)
                predictions = outputs.argmax(dim=1)
                selections = predictions == labels
                samples = inputs[selections]
                datai = torch.cat([datai, samples])
                if datai.shape[0] >= self.MAX_NUM_SAMPLES:
                    break
            num_samples = datai.shape[0]
            print('Number of samples used for generating basis:', num_samples)
            for index in range(len(self.PCA_INDEX) - 1):
                G = datai.view(num_samples,-1)
                G -= G.mean(dim=0, keepdim=True)
                # Compute Principle axis, Principle components, ranks
                Princ_axis, _, _ = SVD(G, self.THD)
                self.Princ_vec[self.PCA_INDEX[index]] = Princ_axis
                
                datai = self.seq_model[self.PCA_INDEX[index]: self.PCA_INDEX[index + 1]](datai)
            G = datai.view(num_samples,-1)
            G -= G.mean(dim=0, keepdim=True)
            # Compute Principle axis, Principle components, ranks
            Princ_axis, _, _ = SVD(G, self.THD)
            self.Princ_vec[self.PCA_INDEX[-1]] = Princ_axis
            
    def from_basis_projection(self):
        # Compute projection matrices from given basis
        for index in range(len(self.PCA_INDEX)):
            basis = self.Princ_vec[self.PCA_INDEX[index]]
            projection = torch.mm(basis, basis.t())
            self.Proj[self.PCA_INDEX[index]] = projection
    
    # Testing linear defense against any selected perturbation
    # defense = None: No defense is applied
    # defense = layer_wise_projection: It projects all input and hidden states onto
    # the range of given projection matrices
    # defense = pmp: Performing PMP iterations
    def testing(self, data_loader, eps, step_size=0., attack='None', defense='None', lr=0., max_iter=0., device=None):
        criterion = nn.CrossEntropyLoss()
        total_step = len(data_loader)
        test_loss = 0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            # Perturbing the input images
            if attack == 'None':
                images_ = inputs
            elif attack == 'fgsm':
                images_ = fgsm(inputs, labels, eps, criterion, self.model)
            elif attack == 'random':
                images_ = random(inputs, labels, eps, criterion, self.model)
            elif attack == 'pgd':
                images_ = pgd(self.model, inputs, labels, criterion, num_steps=20, step_size=step_size, eps=eps)
            elif attack == 'cw':
                print('Processing CW attack on batch:', i)
                CW = CW_attack(self.model)
                images_ = CW.attack(inputs, labels, eps)
            # Defense the input images
            if defense == 'None':
                outputs = self.model(images_)
            elif defense == 'layer_wise_projection':
                outputs = self.Layer_wise_projection(images_)
            elif defense == 'pmp':
                outputs = self.PMP(images_, learning_rate=lr, radius=0., max_iterations=max_iter)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100.*correct/total
        print('Testing accuracy:', accuracy)
        return accuracy
    
    # Testing on the manifold-based attack
    def manifold_attack(self, data_loader, eps, basis, defense='none', device=None):
        criterion = nn.CrossEntropyLoss()
        total_step = len(data_loader)
        test_loss = 0
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            print('Processing CW attack on batch:', i)
            Man_attack = Manifold_attack(model, basis)
            images_ = Man_attack.attack(inputs, labels, eps)
            # Defense the input images
            if defense == 'none':
                outputs = self.model(images_)
            elif defense == 'layer_wise_projection':
                outputs = self.Layer_wise_projection(images_)
            elif defense == 'pmp':
                outputs = self.PMP(images_)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100.*correct/total
        print('Testing accuracy:', accuracy)
        return accuracy
    
    # Projecting all input and hidden states onto the range of
    # given projection matrices
    def Layer_wise_projection(self, data):
        with torch.no_grad():
            for index in range(len(self.PCA_INDEX) - 1):
                data_reconstructed = PCA_Projection(data, self.Proj[self.PCA_INDEX[index]])
                data = self.seq_model[self.PCA_INDEX[index]:self.PCA_INDEX[index + 1]](data_reconstructed)
            data_reconstructed = PCA_Projection(data, self.Proj[self.PCA_INDEX[-1]])
            outputs = self.seq_model[self.PCA_INDEX[-1]:](data)
        return outputs

    # The PMP control algorithm
    # Optimal choices of learning rate and maximum iterations are required //
    # for different neural network and embedding functions
    def PMP(self, data, learning_rate, radius, max_iterations):
        batch_size = data.shape[0]
        data_ = data.clone()
        criterion = nn.MSELoss()
        loss_history = 9999.
        # Initialization of all controls
        Conts = []
        for index in range(len(self.PCA_INDEX) - 1):
            Conts.append(torch.zeros_like(data_, requires_grad=True))
            data_ = self.seq_model[self.PCA_INDEX[index]: self.PCA_INDEX[index + 1]](data_)
        Conts.append(torch.zeros_like(data_, requires_grad=True))
        optimizer = torch.optim.Adam(Conts, learning_rate)
        for ii in range(max_iterations):
            data_prop = data.detach().clone()
            loss = torch.tensor(0.)
            for jj in range(len(self.PCA_INDEX)):
                data_Conted = data_prop + Conts[jj]
                reconstruction = PCA_Projection(data_Conted, self.Proj[self.PCA_INDEX[jj]])
                # Reconstruction loss
                loss1 = criterion(reconstruction, data_Conted)
                
                # Control regularization loss
                # cont_ = Conts[jj].view(batch_size, -1)
                # loss2 = self.C_REGULARIZATION * (cont_.norm(p=2, dim=1)**2).mean()
                # loss = loss + (loss1 + loss2)
                loss = loss + loss1
                
                if jj != (len(self.PCA_INDEX)-1):
                    data_prop = self.seq_model[self.PCA_INDEX[jj]:self.PCA_INDEX[jj+1]](data_Conted)
            if loss.data < loss_history:
                loss_history = loss.data
                data_output = data_Conted

                # Save the final control params that were used for the prediction
                self.last_Conts = [c.clone().detach() for c in Conts]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()  
        outputs = self.seq_model[self.PCA_INDEX[-1]:](data_output)
        return outputs

    # The PMP_testing function uses the first data batch to search for the optimal learning rate //
    # and maximum iterations. 
    # The performance is tested against standard data, fgsm, pgd, cw attacks
    # The function outputs a table consisted of performanc against all perturbations, all magnitudes, //
    # the optimal hyper-parameters are chosen as the pair that corresponds to the best performance    
    def PMP_testing(self, data_loader):
        criterion = nn.CrossEntropyLoss()
        total_step = len(data_loader)
        lr_list = [0.01, 0.0075, 0.005, 0.0025, 0.001]
        iterations_list = [5, 10, 20, 30, 40]
        attack_methods = ['none', 'random', 'fgsm', 'pgd', 'cw']
        data_loader_ = iter(data_loader)
        inputs, labels = next(data_loader_)
        accu = []
        for lr in lr_list:
            for iterations in iterations_list:
                print('Learning rate is:', lr, 'Iterations is:', iterations)
                accu_ = []
                accu_avg = []
                for attack in attack_methods:
                    # print('Attack method:', attack)
                    eps_attack = [2, 4, 8]
                    accuracies = []
                    for eps in eps_attack:
                        if attack == 'random': 
                            data = random(inputs, labels, eps=eps, loss_function=criterion, model=self.model)
                        elif attack == 'fgsm':
                            data = fgsm(inputs, labels, eps=eps, loss_function=criterion, model=self.model)
                        elif attack == 'pgd':
                            data = pgd(self.model, inputs, labels, criterion, num_steps=20, step_size=eps/4, eps=eps)
                        elif attack == 'cw':
                            CW = CW_attack(self.model)
                            data = CW.attack(inputs, labels, eps)
                        elif attack == 'none':
                            data = inputs
                        outputs = self.PMP(data, lr, radius=1000, max_iterations=iterations)

                        _, predicted = outputs.max(1)
                        total = labels.size(0)
                        correct = predicted.eq(labels).sum().item()
                        accuracy = 100.*correct/total
                        accuracies.append(accuracy)
                    accu_.append('{:.1f} / {:.1f} / {:.1f}'.format(accuracies[0], accuracies[1], accuracies[2]))
                    accu_avg.append(accuracies[0] / 3 + accuracies[1] / 3 + accuracies[2] / 3)
                    #print('Accuracy of the network eps=2: {} %'.format(accuracies[0]), 'eps=8: {}'.format(accuracies[1]), 'eps=16: {}'.format((accuracies[2])))
                accu.append(['{} / {}'.format(lr, iterations), accu_[0], accu_[1], accu_[2], accu_[3], accu_[4],
                             (accu_avg[0] + accu_avg[1] + accu_avg[2] + accu_avg[3] + accu_avg[4]) / 5 ])
        table = tabulate(accu, headers=['hyper', 'none', 'random', 'fgsm', 'pgd', 'cw', 'avg'])
        print(table)        
