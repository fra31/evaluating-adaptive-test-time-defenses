#!/bin/bash

# model -> the model from robustbench to use
# indices -> range of images to test


dataset='cifar10'

data_path=" --data_dir /scratch/datasets/CIFAR10"
imgs_path=' --data_path <path>' # path to images to test -- if None clean images are used

#n_ex=1000
#bs=200


# run apgd+eot on the full defense (using the standard model from robustbench as base classifier)
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset cifar10 $data_path --n_ex 1000 --batch_size 45 --model_dir ../../robustbench-addmodels/robustbench/models --config ./configs/cifar10_bpda_eot_sigma025_eot15.yml --eot_test 200 --attack apgd --loss cw --n_iter 10 --sigma_def .25 --max_iters_def 10 --model Standard --indices 0-45

# test points with the full defense
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --model Standard --n_ex 1000  --dataset cifar10 $data_path --model_dir ../../robustbench-addmodels/robustbench/models --config ./configs/cifar10_bpda_eot_sigma025_eot15.yml --eot_test 5 --batch_size 1000 --attack test_points  --sigma_def .25 --max_iter 10 $imgs_path



wait