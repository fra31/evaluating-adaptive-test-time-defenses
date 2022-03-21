#!/bin/bash

data_path=' --data_dir /scratch/datasets/CIFAR10'
n_ex=200
model_name='Andriushchenko2020Understanding' #'Carmon2019Unlabeled' #'Gowal2020Uncovering_28_10_extra' #'Andriushchenko2020Understanding'
bs=200
model_dir='../../robustbench-addmodels/robustbench/models/' 


# apgd on the base model
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset cifar10 --n_ex $n_ex $data_path --model_name $model_name --hd_iter 0 --attack apgd --loss dlr-targeted --n_iter 50 --n_restarts 5 --model_dir $model_dir --eot_iter 0 --batch_size $bs

# apgd + bpda + eot on the defended (5 steps) model
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset cifar10 --n_ex $n_ex $data_path --model_name $model_name --hd_iter 5 --attack apgd --loss dlr-targeted --n_iter 50 --n_restarts 5 --model_dir $model_dir --eot_iter 3 --hd_rs --batch_size $bs

wait

# compute worst-case acc for full defense
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset cifar10 --n_ex $n_ex --model_name $model_name --hd_iter 20 --attack find_wc  --model_dir $model_dir  --batch_size $bs --hd_rs $data_path