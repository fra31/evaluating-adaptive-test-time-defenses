import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

MEAN = torch.tensor([0.485, 0.456, 0.406])
SIGMA = torch.tensor([0.229, 0.224, 0.225])
upper, lower = (1. - MEAN) / SIGMA, (0. - MEAN) / SIGMA
lower = lower[None, :, None, None]
upper = upper[None, :, None, None]

# Project gradient descent attack without maximum epsilon constraint
def pgd_unconstrained(model, x, y, loss_fn, num_steps, step_size):
    step_size = (step_size / 255.) / SIGMA
    step_size = step_size[None, :, None, None]
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)
    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            adv_gradient = _x_adv.grad.sign()
            adv_gradient *= step_size
            x_adv += adv_gradient
            x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach()

# The fast gradient sign method
def fgsm(x, labels, eps, loss_function, model):
    eps = (eps / 255.) / SIGMA
    eps = eps[None, :, None, None]
    
    x_ = x.clone().detach().requires_grad_(True).to(x.device)
    outputs = model(x_)
    loss = loss_function(outputs, labels)
    loss.backward()
    with torch.no_grad():
        adv_gradient = torch.sign(x_.grad)
        adv_gradient *= eps
        x_adv = x_ + adv_gradient
        x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach()

# Uniform random perturbation
def Random(x, labels, eps, loss_function, model):
    eps = (eps / 255.) / SIGMA
    with torch.no_grad():
        adv_gradient = torch.sign(torch.normal(torch.zeros_like(x), torch.ones_like(x)))
        adv_gradient *= eps[None, :, None, None]
        x_adv = x + adv_gradient
        x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach()

# The project gradient descent attack
def pgd(model, x, y, loss_fn, num_steps, step_size, eps):
    step_size = (step_size / 255.) / SIGMA
    step_size = step_size[None, :, None, None]
    eps = (eps / 255.) / SIGMA
    eps = eps[None, :, None, None]
    x_adv = x.clone().to(x.device)
    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)
        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        with torch.no_grad():
            adv_gradient = _x_adv.grad.sign()
            adv_gradient *= step_size
            x_adv += adv_gradient
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.max(torch.min(x_adv, upper), lower)
    return x_adv.detach()

def img_transform(x):
    # Transform images from [0., 1.] range into scaled range
    x -= MEAN[None, :, None, None]
    x /= SIGMA[None, :, None, None]
    return x

def img_de_transform(x):
    # Transform images from scaled range into [0., 1.] range
    x *= SIGMA[None, :, None, None]
    x += MEAN[None, :, None, None]
    return x
    
DECREASE_FACTOR = 0.9  # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 100  # number of iterations to perform gradient descent
ABORT_EARLY = True  # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-5  # the first value of c to start at
LEARNING_RATE = 5e-3  # larger values converge faster to less accurate results
LARGEST_CONST = 0.001  # the largest value of c to go up to before giving up
REDUCE_CONST = False  # try to lower c each iteration; faster to set to false
TARGETED = False  # should we target one specific class? or just be wrong?
CONST_FACTOR = 100.0  # f>1, rate at which we increase constant, smaller better

class CW_attack:
    def __init__(self, model, targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, decrease_factor=DECREASE_FACTOR,
                 const_factor=CONST_FACTOR):
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.num_classes = 10
    
    def attack(self, data, labels, eps):
        r = []
        eps = (eps / 255.) / SIGMA
        eps = eps[None, :, None, None]
        for count, (x, target) in enumerate(zip(data, labels)):
            # Image in scaled range
            x = x.view(1, 3, 32, 32)
            img_ = x.clone()
            # Transform image into range of [-0.5, 0.5]
            img_ = img_de_transform(img_) - 0.5
            x_adv = self.attack_single(img_, target)
            x_adv = img_transform((x_adv + 0.5))
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.max(torch.min(x_adv, upper), lower)        
            
            r.extend(x_adv)
        return torch.stack(r)             
     
    def attack_single(self, img, target):
        prev = img.clone()
        tau = torch.tensor(1.)
        const = self.INITIAL_CONST
        
        while tau > 1 / 256.:
            # try to solve given this tau value
            res = self.gradient_descent(img, target, prev, tau, const)
            if res == None:
                return prev
            
            nimg = res
            if self.REDUCE_CONST: const /= 2
            
            # the attack succeeded, reduce tau and try again
            actualtau = torch.max(np.abs(nimg - img))
            if actualtau < tau:
                tau = actualtau
            prev = nimg
            tau *= self.DECREASE_FACTOR
        return prev 

    def gradient_descent(self, timg, labels, starts, tau, const):
        def atanh(x):
            # Inverse of tanh
            return 0.5*torch.log((1+x)/(1-x))
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        # timg in range of [-0.5, 0.5]
        modifier = torch.zeros_like(timg, dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], self.LEARNING_RATE)
        orig_output = self.model(img_transform(timg + 0.5)).squeeze() 
        # convert to tanh-space
        simg = atanh(starts * 1.999999)
        target_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).type(torch.float32)
        while const < self.LARGEST_CONST:
            for step in range(self.MAX_ITERATIONS):
                # newimg in range of [-0.5, 0.5], perturbation is in inverse tanh space
                newimg = torch.tanh(modifier + simg) / 2                
                # Transform image into scaled range as model input
                output = self.model(img_transform((newimg + 0.5))).squeeze()
                
                real = (target_onehot * output).sum()
                other = ((1 - target_onehot) * output - (target_onehot * 10000)).max()
                
                if self.TARGETED:
                    loss1 = torch.max(torch.zeros(1).to(device), other - real)
                else:
                    loss1 = torch.max(torch.zeros(1).to(device), real - other)

                loss2 = torch.max(torch.zeros(1).to(device), torch.abs(newimg - timg) - tau).sum()
                loss = const * loss1 + loss2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if loss < 0.0001 * const and self.ABORT_EARLY:
                    works = compare(output.argmax(), labels)
                    if works:
                        return newimg.detach()
            
            # we didn't succeed, increase constant and try again
            const *= self.const_factor


# The manifold-based attack is implemented based on the CW attack
# The perturbation is restricted within the embedding sub-space
# Since we consider differentiable data embedding structure (e.g. PCA)
# The CW framefork is a very strong attack candicate
# The following implementaion is based on linear embedding
# Non-linear embedding (e.g. auto-encoder) can be employed
class Manifold_attack:
    def __init__(self, model, basis, targeted=TARGETED, learning_rate=LEARNING_RATE,
                 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY,
                 initial_const=INITIAL_CONST, largest_const=LARGEST_CONST,
                 reduce_const=REDUCE_CONST, decrease_factor=DECREASE_FACTOR,
                 const_factor=CONST_FACTOR):
        self.model = model
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        self.BASIS = basis
        # Get the dimension of embedding sub-space
        _, self.sub_dim = self.BASIS.shape
        self.num_classes = 10
    
    def attack(self, data, labels, eps):
        r = []
        eps = (eps / 255.) / SIGMA
        eps = eps[None, :, None, None]
        for count, (x, target) in enumerate(zip(data, labels)):
            # Image in scaled range
            x = x.view(1, 3, 32, 32)
            img_ = x.clone()
            # Transform image into range of [-0.5, 0.5]
            img_ = img_de_transform(img_) - 0.5
            x_adv = self.attack_single(img_, target)
            x_adv = img_transform((x_adv + 0.5))
            x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            x_adv = torch.max(torch.min(x_adv, upper), lower)        
            
            r.extend(x_adv)
        return torch.stack(r)             
     
    def attack_single(self, img, target):
        prev = img.clone()
        tau = torch.tensor(1.)
        const = self.INITIAL_CONST
        
        while tau > 1 / 256.:
            # try to solve given this tau value
            res = self.gradient_descent(img, target, prev, tau, const)
            if res == None:
                return prev
            
            nimg = res
            if self.REDUCE_CONST: const /= 2
            
            # the attack succeeded, reduce tau and try again
            actualtau = torch.max(np.abs(nimg - img))
            if actualtau < tau:
                tau = actualtau
            prev = nimg
            tau *= self.DECREASE_FACTOR
        return prev 

    def gradient_descent(self, timg, labels, starts, tau, const):
        def atanh(x):
            # Inverse of tanh
            return 0.5*torch.log((1+x)/(1-x))
        def compare(x, y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        # timg in range of [-0.5, 0.5]
        modifier = torch.zeros([self.sub_dim, 1], dtype=torch.float32, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], self.LEARNING_RATE)
        orig_output = self.model(img_transform(timg + 0.5)).squeeze() 
        simg = starts
        target_onehot = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).type(torch.float32)
        while const < self.LARGEST_CONST:
            for step in range(self.MAX_ITERATIONS):
                # Restricting perturbation within the embedding sub-space
                noise = torch.mm(self.BASIS, modifier)
                noise = noise.reshape(timg.shape)
                newimg = simg + noise               
                # Transform image into scaled range as model input
                output = self.model(img_transform((newimg + 0.5))).squeeze()
                
                real = (target_onehot * output).sum()
                other = ((1 - target_onehot) * output - (target_onehot * 10000)).max()
                
                if self.TARGETED:
                    loss1 = torch.max(torch.zeros(1).to(device), other - real)
                else:
                    loss1 = torch.max(torch.zeros(1).to(device), real - other)

                loss2 = torch.max(torch.zeros(1).to(device), torch.abs(newimg - timg) - tau).sum()
                loss = const * loss1 + loss2
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if loss < 0.0001 * const and self.ABORT_EARLY:
                    works = compare(output.argmax(), labels)
                    if works:
                        return newimg.detach()
            
            # we didn't succeed, increase constant and try again
            const *= self.const_factor
