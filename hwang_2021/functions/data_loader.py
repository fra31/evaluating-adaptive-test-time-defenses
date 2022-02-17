import torch
import torchvision
from torchvision import transforms


def dataloader(args, is_preprocessing=True):
    if args.dataset == 'cifar10':
        trainloader, testloader = cifar10_dataloader(args, is_preprocessing=is_preprocessing)
    elif args.dataset == 'tiny':
        trainloader, testloader = tiny_dataloader(args, is_preprocessing=is_preprocessing)
    elif args.dataset == 'cifar100':
        trainloader, testloader = cifar100_dataloader(args, is_preprocessing=is_preprocessing)
    elif args.dataset == 'svhn':
        trainloader, testloader = svhn_dataloader(args)
    else:
        raise print('dataset name error!')

    return trainloader, testloader

def cifar10_dataloader(args, is_preprocessing=True):
    if is_preprocessing:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR10(root=args.dataset_path,
                                            train=True,
                                            download=True,
                                            transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=args.dataset_path,
                                           train=False,
                                           download=True,
                                           transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, testloader

def cifar100_dataloader(args, is_preprocessing=True):
    if is_preprocessing:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR100(root=args.dataset_path,
                                             train=True,
                                             download=True,
                                             transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=args.dataset_path,
                                            train=False,
                                            download=True,
                                            transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, testloader

def tiny_dataloader(args, is_preprocessing):
    if is_preprocessing:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = torchvision.datasets.ImageFolder(args.dataset_path + '/tiny-imagenet-200/train', transform=transform_train)
    testset = torchvision.datasets.ImageFolder(args.dataset_path + '/tiny-imagenet-200/val', transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50, shuffle=False, num_workers=4, pin_memory=True)

    return trainloader, testloader

def svhn_dataloader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.SVHN(root=args.dataset_path, split='train', download=True,
                                         transform=transform_train)
    testset = torchvision.datasets.SVHN(root=args.dataset_path, split='test', download=True,
                                        transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    return trainloader, testloader
