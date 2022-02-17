import os

import torch

from functions import base_function, utils
from functions import data_loader
from functions import train_function
from functions.argparse_function import argparser_function

args = argparser_function()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

utils.set_seed(args.random_seed)

trainloader, testloader = data_loader.dataloader(args=args)

target_model = base_function.load_target_model(dataset=args.dataset, args=args, device=device)

discriminator = base_function.load_discriminator(args=args, target_model=target_model, device=device)


for epoch in range(0, args.max_epoch):
    if args.train_or_test == 'train':
        print('Start training!!')
        print('\nEpoch: %d' % epoch)
        discriminator.train()
        target_model.eval()
        train_function.model_train(discriminator=discriminator,
                                   target_model=target_model,
                                   trainloader=trainloader,
                                   device=device,
                                   args=args)
        base_function.makedirectory(os.path.join(args.pth_path, 'purifier_models', args.dataset))
        torch.save(discriminator.state_dict(), os.path.join(args.pth_path, 'purifier_models', args.dataset, args.main_classifier + '.pt'))
    elif args.train_or_test == 'test':
        print('Load discriminator!!')
        discriminator.load_state_dict(torch.load(os.path.join(args.pth_path, 'purifier_models', args.dataset, args.main_classifier + '.pt'), map_location=device))
    else:
        raise print('train_or_test : train , test option is needed')



attack_list = ['clean', 'pgd', 'cw_l2', 'deepfool', 'mim']
for attack_name in attack_list:
    discriminator.eval()
    target_model.eval()
    _, _ = train_function.model_test(discriminator=discriminator,
                                     target_model=target_model,
                                     testloader=testloader,
                                     attack=attack_name,
                                     device=device,
                                     args=args
                                     )





