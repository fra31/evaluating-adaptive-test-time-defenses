import argparse
import copy
import logging
import os
import time
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preactresnet import PreActResNet18
from wideresnet import WideResNet
from utils_plus import (upper_limit, lower_limit, std, clamp, get_loaders,
    attack_pgd, evaluate_pgd, evaluate_standard, normalize)
from autoattack import AutoAttack
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--data-dir', default='../cifar-data', type=str)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--out-dir', default='train_fgsm_output', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    return parser.parse_args()


def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    logger = logging.getLogger(__name__)

    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler()
        ])

    logger.info(args)

#     _, test_loader = get_loaders(args.data_dir, args.batch_size)
    train_loader, test_loader, train_loader__, test_dataset = get_loaders(args.data_dir, args.batch_size)
    
    best_state_dict = torch.load('./EXP/pre_train_model/wide10_trades_eps8_tricks.pt')
    # Evaluation
    model_test = WideResNet(34, 10, widen_factor=10, dropRate=0.0).cuda()
    if 'state_dict' in best_state_dict.keys():
        model_test.load_state_dict(best_state_dict['state_dict'])
    else:
        model_test.load_state_dict(best_state_dict)
    model_test.float()
    model_test.eval()

    ### Evaluate AutoAttack ###
#     l = [x for (x, y) in test_loader]
#     x_test = torch.cat(l, 0)
#     l = [y for (x, y) in test_loader]
#     y_test = torch.cat(l, 0)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)   
    
        
#     x_test = x_test[2048:2048+512,...]
#     y_test = y_test[2048:2048+512,...]
    
    iii = 19 ##### here we split the set to multi servers and gpus to speed up the test.
             ##### otherwise it is too slow.
    x_test = x_test[256*iii:256*(iii+1),...]
    y_test = y_test[256*iii:256*(iii+1),...]
    
    
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            return self.model_test(normalize(x))
#     new_model = normalize_model(model_test) ### for WideResNet no pre-normalize
    new_model = model_test
    

    epsilon = 8 / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')

#     epsilon = 0.5
#     adversary = AutoAttack(new_model, norm='L2', eps=epsilon, version='standard')


    # adversary.attacks_to_run = ['fab']
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    #     adversary.attacks_to_run = ['apgd-ce', 'apgd-t', 'square']
    # adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
#     adversary.attacks_to_run = ['apgd-t']
    #     adversary.attacks_to_run = ['fab-t','square']
#     adversary.attacks_to_run = ['fab-t']
#     adversary.attacks_to_run = ['square']


    
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
#     X_adv = adversary.run_standard_evaluation_individual(x_test, y_test, bs=256)

if __name__ == "__main__":
    main()
