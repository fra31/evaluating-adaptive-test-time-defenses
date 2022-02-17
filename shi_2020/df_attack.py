import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import numpy as np
import copy
try:
    from torch.autograd.gradcheck import zero_gradients
except:
    pass

class DeepFool():
    r"""
    'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'
    [https://arxiv.org/abs/1511.04599]
    Arguments:
        model (nn.Module): model to attack.
        steps (int): number of steps. (DEFALUT : 3)
        
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
          
    Examples::
        >>> attack = torchattacks.DeepFool(model, steps=3)
        >>> adv_images = attack(images, labels)
        
    """
    def __init__(self, steps=50):
        self.steps = steps

    def __call__(self, model, images):

        # images = images.permute((0,2,3,1))
        adv_images = images.detach().clone()
        for b in range(images.shape[0]):
            r_tot, loop_i, label, k_i, pert_image = deepfool(images[b], model, num_classes=10, overshoot=0.02, max_iter=self.steps)
            # print(pert_image.shape)
            adv_images[b:b+1] = pert_image
        # for b in range(images.shape[0]):

        #     image = images[b:b+1, :, :, :]

        #     image.requires_grad = True
        #     output = model(image)[0]

        #     _, pre_0 = torch.max(output, 0)
        #     f_0 = output[pre_0]
        #     grad_f_0 = torch.autograd.grad(f_0, image,
        #                                    retain_graph=False,
        #                                    create_graph=False)[0]
        #     num_classes = len(output)

        #     for i in range(self.steps):
        #         image.requires_grad = True
        #         output = model(image)[0]
        #         _, pre = torch.max(output, 0)

        #         if pre != pre_0:
        #             image = torch.clamp(image, min=0, max=1).detach()
        #             break

        #         r = None
        #         min_value = None

        #         for k in range(num_classes):
        #             if k == pre_0:
        #                 continue

        #             f_k = output[k]
        #             grad_f_k = torch.autograd.grad(f_k, image,
        #                                            retain_graph=True,
        #                                            create_graph=True)[0]

        #             f_prime = f_k - f_0
        #             grad_f_prime = grad_f_k - grad_f_0
        #             value = torch.abs(f_prime)/torch.norm(grad_f_prime)

        #             if r is None:
        #                 r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
        #                 min_value = value
        #             else:
        #                 if min_value > value:
        #                     r = (torch.abs(f_prime)/(torch.norm(grad_f_prime)**2))*grad_f_prime
        #                     min_value = value

        #         image = torch.clamp(image + r, min=0, max=1).detach()

        #     images[b:b+1, :, :, :] = image

        # adv_images = images
        # print('1 batch')
        # print(adv_images.shape)
        return adv_images
        # return adv_images.permute((0,3,1,2))


def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """
    # is_cuda = torch.cuda.is_available()

    # if is_cuda:
    #     # print("Using GPU")
    #     image = image.cuda()
    #     net = net.cuda()
    # else:
    #     None
    #     # print("Using CPU")


    f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0]

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        # if is_cuda:
        #     pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        # else:
        #     pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)
        pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(image.device)
        pert_image = pert_image.clamp(0,1)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image