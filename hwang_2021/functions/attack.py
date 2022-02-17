import sys

sys.path.append('..')
from functions.utils import *

def pgd_attack_for_mixed(model, ensemble_model, images, labels, lambda_sra, targeted=False, eps=8/255, alpha=1/255, iters=20, random_start=True):

    loss = nn.CrossEntropyLoss()
    loss_BCE = nn.BCELoss()

    if targeted:
        loss = lambda x, y: -nn.CrossEntropyLoss()(x, y)

    ori_images = images.clone().detach()

    batch_size = images.shape[0]

    targets_joint = make_target(batch_size, 'cuda', half_label=False)

    if random_start:
        # Starting at a uniformly random point
        images = images + torch.empty_like(images).uniform_(-eps, eps)
        images = torch.clamp(images, min=0, max=1)

    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)
        outputs2 = ensemble_model(images)


        cost = loss(outputs, labels) - lambda_sra * loss_BCE(outputs2, targets_joint)

        grad = torch.autograd.grad(cost, images,
                                   retain_graph=False, create_graph=False)[0]

        adv_images = images + alpha * grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach()

    adv_images = images

    return adv_images

def mim_attack(predict, x, y, targeted=False, eps=0.3, nb_iter=40, decay_factor=1., eps_iter=0.003, clip_min=0., clip_max=1., loss_fn=None):
    """
    The L-inf projected gradient descent attack (Dong et al. 2017).
    The attack performs nb_iter steps of size eps_iter, while always staying
    within eps from the initial point. The optimization is performed with
    momentum.
    Paper: https://arxiv.org/pdf/1710.06081.pdf
    """


    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss(reduction="sum")



    if targeted:
        raise print('targeted is not implemented')

    x = x.detach().clone()
    y = y.detach().clone()

    delta = torch.zeros_like(x)
    g = torch.zeros_like(x)

    delta = nn.Parameter(delta, requires_grad=True)

    for i in range(nb_iter):

        if delta.grad is not None:
            delta.grad.detach_()
            delta.grad.zero_()

        imgadv = x + delta
        outputs = predict(imgadv)
        loss = loss_fn(outputs, y)
        if targeted:
            loss = -loss
        loss.backward()

        g = decay_factor * g + normalize_by_pnorm(delta.grad.data, p=1)
        # according to the paper it should be .sum(), but in their
        #   implementations (both cleverhans and the link from the paper)
        #   it is .mean(), but actually it shouldn't matter

        delta.data += eps_iter * torch.sign(g)
        # delta.data += self.eps / self.nb_iter * torch.sign(g)

        delta.data = clamp(
            delta.data, min=-eps, max=eps)
        delta.data = clamp(
            x + delta.data, min=clip_min, max=clip_max) - x

    rval = x + delta.data
    return rval


def _batch_multiply_tensor_by_vector(vector, batch_tensor):
    """Equivalent to the following
    for ii in range(len(vector)):
        batch_tensor.data[ii] *= vector[ii]
    return batch_tensor
    """
    return (
        batch_tensor.transpose(0, -1) * vector).transpose(0, -1).contiguous()

def batch_multiply(float_or_vector, tensor):
    if isinstance(float_or_vector, torch.Tensor):
        assert len(float_or_vector) == len(tensor)
        tensor = _batch_multiply_tensor_by_vector(float_or_vector, tensor)
    elif isinstance(float_or_vector, float):
        tensor *= float_or_vector
    else:
        raise TypeError("Value has to be float or torch.Tensor")
    return tensor

def _get_norm_batch(x, p):
    batch_size = x.size(0)
    return x.abs().pow(p).view(batch_size, -1).sum(dim=1).pow(1. / p)

def normalize_by_pnorm(x, p=2, small_constant=1e-6):
    """
    Normalize gradients for gradient (not gradient sign) attacks.
    # TODO: move this function to utils
    :param x: tensor containing the gradients on the input.
    :param p: (optional) order of the norm for the normalization (1 or 2).
    :param small_constant: (optional float) to avoid dividing by zero.
    :return: normalized gradients.
    """
    # loss is averaged over the batch so need to multiply the batch
    # size to find the actual gradient of each input sample

    assert isinstance(p, float) or isinstance(p, int)
    norm = _get_norm_batch(x, p)
    norm = torch.max(norm, torch.ones_like(norm) * small_constant)
    return batch_multiply(1. / norm, x)

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.clamp(input, min=min, max=max)
    elif min is None and max is None:
        return input
    elif min is None and max is not None:
        return torch.clamp(input, max=max)
    elif min is not None and max is None:
        return torch.clamp(input, min=min)
    else:
        raise ValueError("This is impossible")
