#!/bin/bash

TRAIN_DIR="/home/neni/imagenet/train"
VAL_DIR="/home/neni/imagenet/val"

LOG_PATH="../logs/pytorch/misc_imagenet"

#python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_fast_collate --log_path $LOG_PATH --gpu_profiler --train_path $TRAIN_DIR --test_path $VAL_DIR -bs 128 256 512

#python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_fast_collate_prefetch --log_path $LOG_PATH --gpu_profiler --train_path $TRAIN_DIR --test_path $VAL_DIR -bs 128 256 512

python ../src/train_pytorch.py --name pytorch_resnet18 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --model resnet18 --gpu_profiler --num_workers 8 -bs 256 512 1024 --width 256 --height 256 --crop 224 --num_classes 1000

python ../src/train_pytorch.py --name pytorch_resnet18_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --model resnet18 --gpu_profiler --num_workers 8 --use_dali -bs 256 512 1024 --width 256 --height 256 --crop 224 --num_classes 1000

python ../src/train_pytorch.py --name pytorch_resnet18_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --model resnet18 --gpu_profiler --num_workers 8 --use_dali --dali_cpu -bs 256 512 1024 --width 256 --height 256 --crop 224 --num_classes 1000