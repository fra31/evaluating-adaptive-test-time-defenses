#!/bin/bash

model_dir='./'
data_dir='/scratch/datasets/CIFAR10'

echo 'original points, no purification'
CUDA_VISIBLE_DEVICES=2 python3 eval_tta.py --dataset cifar10  --model_name resnet  --model_dir ./  --eps 8  --pfy_delta 0 --data_dir /scratch/datasets/CIFAR10 --n_ex 1000 --pfy_iter 5  --batch_size 1000 --pfy_step_size 4

echo 'original points, with purification'
CUDA_VISIBLE_DEVICES=2 python3 eval_tta.py --dataset cifar10  --model_name resnet  --model_dir ./  --eps 8  --pfy_delta 4 --data_dir /scratch/datasets/CIFAR10 --n_ex 1000 --pfy_iter 5  --batch_size 1000 --pfy_step_size 4

#

echo 'fgsm (reimpl.) without bpda'
CUDA_VISIBLE_DEVICES=2 python3 eval_tta.py --dataset cifar10  --model_name resnet  --model_dir ./  --eps 8  --pfy_delta 4 --data_dir /scratch/datasets/CIFAR10 --n_ex 1000 --pfy_iter 5  --batch_size 1000 --attack fgsm --save_imgs --pfy_step_size 4

echo 'apgd+bpda (on trajectory)'
CUDA_VISIBLE_DEVICES=2 python3 eval_tta.py --dataset cifar10  --model_name resnet  --model_dir ./  --eps 8  --pfy_delta 4 --data_dir /scratch/datasets/CIFAR10 --n_ex 1000 --pfy_iter 5  --batch_size 1000 --pfy_step_size 4  --attack apgd_tta

echo 'test apgd+bpda points'
CUDA_VISIBLE_DEVICES=2 python3 eval_tta.py --dataset cifar10  --model_name resnet  --model_dir ./  --eps 8  --pfy_delta 4 --data_dir /scratch/datasets/CIFAR10 --n_ex 1000 --pfy_iter 5  --batch_size 1000 --pfy_step_size 4  --data_path ./results/cifar10/Linf/resnet/apgd_tta_1_1000_eps_0.03137_pfyiter_5_dyn_False_loss_ce_2000x1_eot_1_niter_1000.pth

