#!/bin/bash

TRAIN_DIR="/home/neni/tiny-imagenet-200/train"
VAL_DIR="/home/neni/tiny-imagenet-200/val"

LOG_PATH="../logs/pytorch/tiny_imagenet64"

# BASELINE
# resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR -bs 128 256 512
# resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 3 -bs 128 256 512
# resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 8 -bs 128 256 512
# resnet50, synthetic data
python ../src/train_pytorch.py --name pytorch_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --synthetic_data -bs 128 256 512

LOG_PATH="../logs/pytorch_dali_gpu/tiny_imagenet64"

# DALI (GPU)
# resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --use_dali -bs 128 256 512
## resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 3 --use_dali -bs 128 256 512
## resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 8 --use_dali -bs 128 256 512

LOG_PATH="../logs/pytorch_dali_cpu/tiny_imagenet64"

## DALI (CPU)
## resnet50, no optimizations
python ../src/train_pytorch.py --name pytorch_resnet50_no_optim_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --use_dali --dali_cpu -bs 128 256 512
## resnet50, no optimizations. only 3 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 3 --use_dali --dali_cpu -bs 128 256 512
## resnet50, no optimizations. 8 workers
python ../src/train_pytorch.py --name pytorch_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 8 --use_dali --dali_cpu -bs 128 256 512