#!/bin/bash

norm='L2' #'Linf' #'L2'
eps=.5 #8  .5

# models: wrn_at -> base model with TRADES
#         orig_wrn -> TRADES + SODEF

# autoattack on the base model
: 'CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack eval_fast --eot_test 1 --batch_size 1000 --eps $eps --model wrn_at --norm $norm

# apgd on the base model
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack apgd --eot_test 1 --batch_size 1000 --eps $eps --model wrn_at --norm $norm --loss ce --n_iter 100 --n_restarts 1
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack apgd --eot_test 1 --batch_size 1000 --eps $eps --model wrn_at --norm $norm --loss cw --n_iter 100 --n_restarts 1
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack apgd --eot_test 1 --batch_size 1000 --eps $eps --model wrn_at --norm $norm --loss dlr-targeted --n_iter 100 --n_restarts 3
'

# transfer apgd from base model (worst-case)
CUDA_VISIBLE_DEVICES=1 python3 eval_tta.py --n_ex 1000 --data_dir /scratch/datasets/CIFAR10 --attack find_wc --eot_test 1 --batch_size 1000 --eps $eps --model orig_wrn --norm $norm

