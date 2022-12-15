#!/bin/bash

TRAIN_DIR="/home/neni/imagenet/train"
VAL_DIR="/home/neni/imagenet/val"

LOG_PATH="../logs/pytorch/imagenet_200k"

# resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, synthetic data
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --synthetic_data -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000

LOG_PATH="../logs/pytorch_dali_gpu/imagenet_200k"

# DALI (GPU)
# resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
## resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
## resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000

LOG_PATH="../logs/pytorch_dali_cpu/imagenet_200k"

## DALI (CPU)
## resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
## resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
## resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000