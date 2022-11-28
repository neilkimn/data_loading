#!/bin/bash

TRAIN_DIR="/home/neni/tiny-imagenet-200/train"
VAL_DIR="/home/neni/tiny-imagenet-200/val"

# BASELINE
# resnet50, no optimizations
#python ../src/train_pytorch.py --name pytorch_resnet50_no_optim --log_path ../logs/pytorch/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler
# resnet50, no optimizations. only 3 workers
#python ../src/train_pytorch.py --name pytorch_resnet50 --log_path ../logs/pytorch/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 3 --gpu_profiler
# resnet50, no optimizations. 8 workers
#python ../src/train_pytorch.py --name pytorch_resnet50 --log_path ../logs/pytorch/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 8 --gpu_profiler
# resnet50, synthetic data
#python ../src/train_pytorch.py --name pytorch_resnet50 --log_path ../logs/pytorch/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --synthetic_data

# DALI (GPU)
# resnet50, no optimizations
#python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_GPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali
## resnet50, no optimizations. only 3 workers
#python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali
## resnet50, no optimizations. 8 workers
#python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali

## DALI (CPU)
## resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_CPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu
## resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu
## resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path ../logs/pytorch_dali/imagenet64_100k/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --dali_cpu