import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# import foolbox as fb

from datasets import *
from utils import *
from attacks import *
from defenses import *
from models import *
from criterions import *

import argparse
import os
import sys
import json
import time
from collections import deque
from functools import partial
from datetime import datetime

def train_main(args):

    torch.manual_seed(0)

    # path initialization
    if args.name is None:
        args.name = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    save_dir = os.path.join(args.save_dir, args.dataset, args.name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if os.path.exists('{}/latest_model.pth'.format(save_dir)) and not args.pretrained:
        print('model exists!')
        return

    # redirect output
    file = open(os.path.join(save_dir, 'train_log.txt'), 'a')
    sys.stdout = file

    # load data
    if args.auxiliary == 'pi':
        train_transform = load_transform(args.dataset, mod='test') # data augmentation is wrapped in the auxiliary loss function
    else:
        train_transform = load_transform(args.dataset, mod='train')
    test_transform = load_transform(args.dataset, mod='test')
    print('training transformations: ', train_transform)
    
    data_dir = '/home/scratch/datasets/CIFAR10'
    train_set = load_dataset(name=args.dataset, mod='train', transform=train_transform, root_dir=data_dir)
    test_set = load_dataset(name=args.dataset, mod='test', transform=test_transform, root_dir=data_dir)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    # parallelism
    if args.device == 'cuda':
        if args.gpus:
            device_ids = [int(idx) for idx in list(args.gpus)]
            device = '{}:{}'.format(args.device, device_ids[0])
        else:
            device = args.device
            device_ids = [0]
    elif args.device == 'cpu':
        device = args.device

    # load model
    in_channel = 1 if args.dataset == 'mnist' or args.dataset == 'fmnist' else 3
    n_class = 100 if args.dataset == 'cifar100' else 10
    if args.pretrained:
        model = load_model(args.model, args.dataset, n_class, in_channel, save_dir='./results/cifar10/2021-07-28-12:32:31').to(device) #save_dir
    else:
        model = load_model(args.model, args.dataset, n_class, in_channel).to(device)

    # optimizer initialization
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=args.lr_gamma)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    print('optim: ', optimizer)
    print('schedule: ', scheduler)

    # attacks intialization
    args.epsilon = args.epsilon / 255 if args.epsilon > 1 else args.epsilon
    args.step_size = args.step_size / 255 if args.step_size > 0.1 else args.step_size
    if args.attack is not None:
        if args.attack == 'fgsm':
            attack = partial(globals()[args.attack], epsilon=args.epsilon)
        elif args.attack == 'pgd_linf':
            attack = partial(globals()[args.attack], epsilon=args.epsilon, alpha=args.step_size, num_iter=args.num_iter)
    else:
        attack = None
    print('attack: ', args.attack, '-', args.epsilon, ', ', args.step_size, '*', args.num_iter)

    # criterion initialization
    if args.auxiliary == 'rec':
        criterion = partial(joint_criterion, aux_criterion=recon_criterion, alpha=args.alpha)
    elif args.auxiliary == 'pi':
        criterion = partial(joint_criterion, aux_criterion=pi_criterion, alpha=args.alpha)
    elif args.auxiliary == 'rot':
        criterion = partial(joint_criterion, aux_criterion=rotate_criterion, alpha=args.alpha)
    else:
        criterion = nn.CrossEntropyLoss()
    print('auxiliary: ', args.auxiliary, ' - alpha = {}'.format(args.alpha))

    # model training
    print('Start training...')
    T = []
    for epoch in range(args.epochs):
        print('epoch: {}'.format(epoch))
        t0 = time.time()
        if args.auxiliary is not None:
            train_with_auxiliary(model, train_loader, criterion, optimizer, scheduler, device)
        else:
            train_adversarial(model, train_loader, criterion, attack, optimizer, device)
        T.append(time.time()-t0)
        scheduler.step()
        # model validation
        if epoch % args.validate_freq == 0:
            evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
        if epoch % args.save_freq == 0:
            torch.save(model.state_dict(), '{}/latest_model_{}.pth'.format(save_dir, epoch))

    torch.save(model.state_dict(), '{}/latest_model.pth'.format(save_dir))

    model.eval()
    evaluate(model, train_loader, nn.CrossEntropyLoss(), device)
    evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    file.close()

def train_sub(args):

    save_dir = os.path.join(args.save_dir, args.dataset, args.name)

    # redirect output
    file = open(os.path.join(save_dir, '{}_sub_train_log.txt'.format(args.sub_model)), 'a')
    sys.stdout = file

    print(args.note)

    # import targeted model
    in_channel = 1 if args.dataset == 'mnist' or args.dataset == 'fmnist' else 3
    tar_model = load_model(args.model, args.dataset, in_channel, save_dir=save_dir)
    tar_model.eval()

    # parallelism
    if args.device == 'cuda':
        if args.gpus:
            device_ids = [int(idx) for idx in list(args.gpus)]
            device = '{}:{}'.format(args.device, device_ids[0])
        else:
            device = args.device
            device_ids = [0]
    elif args.device == 'cpu':
        device = args.device

    # load partial holdout data
    transform = load_transform(args.dataset, mod='test')
    test_set = load_dataset(name=args.dataset, mod='test', transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    data = test_set.data[:150]

    print('lr: ', args.lr)

    # model training
    print('Start training...')
    args.epochs = 10
    for sub_epoch in range(args.sub_epochs):

        print('substitute epoch: {}'.format(sub_epoch))

        if sub_epoch == 0:
            data = test_set.data[:150].unsqueeze(dim=1).float()/255
            y = tar_model(data).argmax(dim=1)
        else:
            lmbda = (2 * int(int(sub_epoch / 3) == 0) - 1) * 0.1
            print('lambda: ', lmbda)
            new_data = jacobian_augment(model, train_loader, lmbda, device)
            data = torch.cat([data, new_data], dim=0)
            y = torch.cat([y, tar_model(new_data).argmax(dim=1)])
        # import substitute model (train from scratch every substitute epoch)
        model = load_model(args.sub_model, in_channel).to(device)

        # optimizer initialization
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)

        dataset = torch.utils.data.TensorDataset(data, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=150, shuffle=True, num_workers=args.workers, drop_last=False)
        
        for epoch in range(args.epochs):
            print('epoch: ', epoch)
            train(model, train_loader, nn.CrossEntropyLoss(), optimizer, scheduler, device)

        # model validation
        if epoch % args.validate_freq == 0:
            evaluate(model, test_loader, nn.CrossEntropyLoss(), device)

    torch.save(model.state_dict(), '{}/substitute_{}.pth'.format(save_dir, args.sub_model))
        

def test_main(args):

    assert args.name is not None
    save_dir = args.name #os.path.join(args.save_dir, args.dataset, args.name)

    # redirect output
    file = open(os.path.join(save_dir, 'test_log.txt'), 'a')
    #sys.stdout = file

    # data loading
    transform = load_transform(args.dataset, mod='test')
    data_dir = '/home/scratch/datasets/CIFAR10'
    #train_set = load_dataset(name=args.dataset, mod='train', transform=transform)
    test_set = load_dataset(name=args.dataset, mod='test', transform=transform, root_dir=data_dir)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, drop_last=False)

    in_channel = 1 if args.dataset == 'mnist' or args.dataset == 'fmnist' else 3
    n_class = 100 if args.dataset == 'cifar100' else 10
    model = load_model(args.model, args.dataset, n_class, in_channel, save_dir=save_dir)

    # criterion & device
    if args.device == 'cuda':
        if args.gpus:
            device_ids = [int(idx) for idx in list(args.gpus)]
            device = '{}:{}'.format(args.device, device_ids[0])
        else:
            device = args.device
            device_ids = [0]
    elif args.device == 'cpu':
        device = args.device

    model = model.to(device)
    model.eval()
        
    # auxiliary task
    if args.auxiliary == 'rec':
        aux_criterion = recon_criterion
    elif args.auxiliary == 'rot':
        aux_criterion = rotate_criterion_l2
    elif args.auxiliary == 'pi':
        aux_criterion = pi_criterion
    else:
        aux_criterion = None

    if args.second_order:
        criterion = partial(second_order, df_criterion=aux_criterion, beta=args.beta)
        print(criterion)
    else:
        criterion = nn.CrossEntropyLoss()

    # attack
    if args.attack != 'cw' and args.attack != 'df':
        args.epsilon = args.epsilon / 255 if args.dataset == 'cifar10' or args.dataset == 'cifar100' else args.epsilon
        args.step_size = args.step_size / 255 if args.dataset == 'cifar10' or args.dataset == 'cifar100' else args.step_size
        args.pfy_delta = args.pfy_delta / 255 if args.dataset == 'cifar10' or args.dataset == 'cifar100' else args.pfy_delta
        args.pfy_step_size = args.pfy_step_size / 255 if args.dataset == 'cifar10' or args.dataset == 'cifar100' else args.pfy_step_size

    if args.attack is not None:
        if args.attack == 'fgsm':
            attack = partial(globals()[args.attack], epsilon=args.epsilon)
        elif args.attack == 'pgd_linf':
            attack = partial(globals()[args.attack], epsilon=args.epsilon, step_size=args.step_size, num_iter=args.num_iter)
        elif args.attack == 'cw':
            attack = partial(globals()[args.attack], epsilon=args.epsilon, num_classes=n_class)
        elif args.attack == 'df':
            attack = partial(globals()[args.attack], epsilon=args.epsilon)
        elif args.attack == 'bpda':
            attack = partial(globals()[args.attack], epsilon=args.epsilon, step_size=args.step_size, num_iter=args.num_iter, purify=partial(globals()['purify'], aux_criterion=aux_criterion))
        elif args.attack == 'black_box':
            # black-box fgsm attack
            sub_model = load_model(args.sub_model, in_channel, save_dir=save_dir, substitute=True).to(device)
            sub_model.eval()
            attack = partial(fgsm, model=sub_model, epsilon=args.epsilon)
        elif args.attack == 'empty':
            attack = partial(globals()[args.attack])

    if args.defense is not None:
        assert args.auxiliary is not None
        pfy = partial(purify, defense_mode=args.defense, delta=args.pfy_delta, step_size=args.pfy_step_size, num_iter=args.pfy_num_iter)
    else:
        defense = None

    if args.attack is not None:
        evaluate_adversarial(model, test_loader, criterion, aux_criterion, attack, pfy, device)
    else:
        evaluate(model, test_loader, criterion, device)

    file.close()


def parse_args():

    parser = argparse.ArgumentParser(description='PyTorch Adversarial Robustness')

    # running and saving
    parser.add_argument('-sd', '--save-dir', default='./results', help='path where to save')
    parser.add_argument('-n', '--name', default=None, help='folder in the save path')
    parser.add_argument('-p', '--pretrained', dest="pretrained", help="Use pre-trained models", action="store_true")
    parser.add_argument('-t', '--test-only', dest="test_only", help="Only test the model", action="store_true")
    parser.add_argument('--sub', help="train substitute model", action="store_true")

    # model and dataset
    parser.add_argument('-m', '--model', default='fcnet', help='type of the model')
    parser.add_argument('-sm', '--sub-model', default='fcnet', help='type of the substitute model')
    parser.add_argument('-d', '--dataset', default='mnist', help='name of the dataset')

    # adversarial attacks
    parser.add_argument('-at', '--attack', default=None, help="adversarial attack strategy")
    parser.add_argument('-e', '--epsilon', type=float, default=0.1, help="adversary epsilon")
    parser.add_argument('-s', '--step-size', type=float, default=0.01, help="pgd adversary step size")
    parser.add_argument('-ni', '--num-iter', type=int, default=40, help="pgd adversary number of iteration")
    parser.add_argument('-so', '--second-order', help="use second order attack", action="store_true")
    parser.add_argument('-b', '--beta', type=float, default=1, help="weight of auxiliary aware adversaries")

    # adversarial defense (purification)
    parser.add_argument('-df', '--defense', default=None, help="adversarial defense strategy")
    parser.add_argument('-aux', '--auxiliary', default=None, help="auxiliary task for defense")
    parser.add_argument('-a', '--alpha', type=float, default=1, help="weight of auxiliary loss during training")
    parser.add_argument('-pd', '--pfy-delta', type=float, default=0.1, help="unit increment when purifier searching for the best purification epsilon")
    parser.add_argument('-ps', '--pfy-step-size', type=float, default=0.1, help="purifier step size")
    parser.add_argument('-pni', '--pfy-num-iter', type=int, default=5, help="purifier number of iteration")
    
    # optimization
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-u', '--gpus', default='', help='index of specified gpus')
    parser.add_argument('-bs', '--batch-size', default=128, type=int)
    parser.add_argument('-ep', '--epochs', default=100, type=int, metavar='N', help='# of total epochs')
    parser.add_argument('-sep', '--sub-epochs', default=6, type=int, metavar='N', help='# of total substitute epochs')
    parser.add_argument('-j', '--workers', default=32, type=int, metavar='N', help='# of data loaders (default: 16)')
    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate (default for sgd: 0.1)')
    parser.add_argument('--lr-step', default=100, type=int, help='decrease lr every these iterations')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M')
    # parser.add_argument('--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--validate-freq', default=1, type=int, help='validation frequency')
    parser.add_argument('--save-freq', default=20, type=int, help='save model frequency')

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    if args.test_only:
        test_main(args)
    else:
        if args.sub:
            train_sub(args)
        else:
            train_main(args)