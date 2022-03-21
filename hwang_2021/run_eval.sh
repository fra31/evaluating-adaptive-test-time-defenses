#!/bin/bash

dataset='svhn' #'cifar100' #'svhn' #'cifar100'
mc='madry' #'lee' #'zhang'

data_path=" --data_dir /scratch/datasets/${dataset^^}"
model_dir=' --pth_path ../../test_time_defenses_eval/aid_purifier'
imgs_dir="../../test_time_defenses_eval/aid_purifier/code_final/results/${dataset}/${mc}"
imgs_path='eval_fast_1_1000_pursteps_10_short.pth'
n_ex=1000
bs=128

echo "starting eval"

if [ $dataset = 'svhn' ]
then
    # autoattack on base model
    #CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset svhn --n_ex $n_ex -tt test -layer2_off -layer3_off -def_eps 0.047 -def_alpha 0.012 -bcd 0 -acd 2 --discr_eval_mode -mc $mc --attack eval_fast --defense_step 0 $data_path --batch_size $bs $model_dir
    
    # apgd+bpda on full defense
    #CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset svhn --n_ex $n_ex -tt test -layer2_off -layer3_off -def_eps 0.047 -def_alpha 0.012 -bcd 0 -acd 2 --discr_eval_mode -mc $mc --attack eval_fast_short --defense_step 10 $data_path --batch_size $bs $model_dir
    
    # higher budget apgd+bpda
    CUDA_VISIBLE_DEVICES=3 python3 eval_tta.py --dataset svhn --n_ex $n_ex -tt test -layer2_off -layer3_off -def_eps 0.047 -def_alpha 0.012 -bcd 0 -acd 2 --discr_eval_mode -mc $mc --attack apgd --defense_step 10 $data_path --batch_size $bs --loss dlr-targeted --n_iter 10 --n_restarts 5 --data_path "${imgs_dir}/${imgs_path}" $model_dir

elif [ $dataset = 'cifar10' ]
then
    # autoattack on base model
    CUDA_VISIBLE_DEVICES=3 python3 eval_tta.py --dataset cifar10 --n_ex $n_ex -tt test -layer2_off -layer4_off -def_eps 0.031 -def_alpha 0.008 -def_step 0 --discr_eval_mode -mc madry --attack eval_fast $data_path $model_dir $model_dir

    # apgd+bpda on full defense
    CUDA_VISIBLE_DEVICES=3 python3 eval_tta.py --dataset cifar10 --n_ex $n_ex -tt test -layer2_off -layer4_off -def_eps 0.031 -def_alpha 0.008 -def_step 10 --discr_eval_mode -mc madry --attack eval_fast_short $data_path $model_dir

    #

else
    # autoattack on base model
    CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset $dataset --n_ex $n_ex -tt test -layer2_off -layer4_off -def_eps 0.062 -def_alpha 0.008 -def_step 0 --discr_eval_mode -mc $mc --attack eval_fast $data_path $model_dir
    
    # apgd+bpda on full defense
    CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset $dataset --n_ex $n_ex -tt test -layer2_off -layer4_off -def_eps 0.062 -def_alpha 0.008 -def_step 10 --discr_eval_mode -mc $mc --attack eval_fast_short $data_path $model_dir
    
    #CUDA_VISIBLE_DEVICES=0 python3 eval_tta.py --dataset $dataset --n_ex $n_ex -tt test -layer2_off -layer4_off -def_eps 0.062 -def_alpha 0.008 -def_step 10 --discr_eval_mode -mc $mc --attack test_points $data_path --data_path "${imgs_dir}/${imgs_path}" $model_dir
    
    
    
fi

wait