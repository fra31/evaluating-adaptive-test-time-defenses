import os

import torch

from purifying_model import discriminator_model, discriminator_model_new
from target_model import wideresnet, small_CNN


def load_target_model(dataset, args, device):
    if dataset == 'cifar10':
        target_model = wideresnet.WideResNet().to(device)
    elif dataset == 'svhn':
        target_model = wideresnet.WideResNet().to(device)
    elif dataset == 'cifar100':
        target_model = wideresnet.WideResNet(num_classes=100).to(device)
    elif dataset == 'tiny':
        target_model = wideresnet.WideResNet(num_classes=200).to(device)
    else:
        raise print('Enter the exact dataset (svhn, cifar10, cifar100, tiny)')

    target_model.load_state_dict(torch.load(os.path.join(args.pth_path, 'main_classification_models', args.dataset, '-'.join(['model', args.main_classifier, 'epoch_last.pt'])), map_location=device))

    target_model.eval()

    for p in target_model.parameters():
        p.requires_grad = False

    return target_model

def parallel_GPU_ON(target_model):
    import torch.backends.cudnn as cudnn

    target_model = torch.nn.DataParallel(target_model)
    cudnn.benchmark = True

    return target_model

def load_discriminator(args, target_model, device):
    if not args.use_custom_discr:
        discriminator = discriminator_model.ConcatDiscriminator(target_model=target_model, args=args).to(device)
    else:
        print('not using the original implementation of the discriminator')
        discriminator = discriminator_model_new.ConcatDiscriminator(target_model=target_model, args=args).to(device)
    
    return discriminator

def makedirectory(str_dir):
    if not os.path.exists(str_dir):
        os.makedirs(str_dir)



#from foolbox.attacks import LinfPGD, L2CarliniWagnerAttack, LinfDeepFoolAttack, FGSM
from functions.attack import mim_attack


def generate_pgd_for_training(fmodel, inputs, targets, args):

    attack_f = LinfPGD(abs_stepsize=args.training_step_size, steps=args.training_step)
    _, adv_data, success = attack_f(fmodel, inputs, targets, epsilons=args.training_eps)

    return adv_data

def generate_attack_images(dataset, attack, fmodel, target_model, inputs, targets):

    if dataset == 'svhn':
        epsilon = 0.047
        alpha = epsilon / 6
    else:
        epsilon = 0.031
        alpha = epsilon / 8

    if attack == 'fgsm':
        attack_f = FGSM()
        _, adv_data, success = attack_f(fmodel, inputs, targets, epsilons=epsilon)
    elif attack == 'cw_l2':
        attack_f = L2CarliniWagnerAttack(binary_search_steps=1, initial_const=0.01, steps=20)
        _, adv_data, success = attack_f(fmodel, inputs, targets, epsilons=3)
    elif attack == 'pgd':
        attack_f = LinfPGD(abs_stepsize=alpha, steps=40)
        _, adv_data, success = attack_f(fmodel, inputs, targets, epsilons=epsilon)
    elif attack == 'deepfool':
        attack_f = LinfDeepFoolAttack(steps=20)
        _, adv_data, success = attack_f(fmodel, inputs, targets, epsilons=epsilon)
    elif attack == 'mim':
        adv_data = mim_attack(target_model, inputs, targets, eps=epsilon)
    elif attack == 'clean':
        adv_data = inputs
    else:
        raise print('attack name error')

    return adv_data


def make_target(batchsize, device, half_label=True):
    if half_label:
        targets = torch.ones((batchsize, 1)).to(device)
        targets[:batchsize//2] = targets[:batchsize//2] * 0
    else:
        targets = torch.ones((batchsize, 1)).to(device)

    return targets
