#!/bin/bash

# model -> the model from robustbench to use
# indices -> range of images to test


dataset='cifar10'

data_path=" --data_dir /scratch/datasets/CIFAR10"
model_dir=' --model_dir ../../robustbench-addmodels/robustbench/models' # path to robustbench models
ebm_dir=' --ebm_dir ../../test_time_defenses_eval/adp' # path ebm model
imgs_path=' --data_path ./results/Standard/apgd_1_1000_niter_10_loss_cw_nrestarts_1_init_None_sigma_0.25_maxiter_10_eotiter_200_0-45_best.pth' # path to images to test - if None clean images are used


# run apgd+eot on the full defense (using the standard model from robustbench as base classifier)
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset cifar10 $data_path --n_ex 1000 --batch_size 45 $model_dir --config ./configs/cifar10_bpda_eot_sigma025_eot15.yml --eot_test 200 --attack apgd --loss cw --n_iter 10 --sigma_def .25 --max_iters_def 10 --model Standard --indices 0-45 $ebm_dir

# test points with the full defense
CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --model Standard --n_ex 1000  --dataset cifar10 $data_path $model_dir --config ./configs/cifar10_bpda_eot_sigma025_eot15.yml --eot_test 5 --batch_size 1000 --attack test_points  --sigma_def .25 --max_iters_def 10 $imgs_path $ebm_dir


wait