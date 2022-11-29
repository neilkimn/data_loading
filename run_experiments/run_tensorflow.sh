#!/bin/bash

TRAIN_DIR="/home/neni/tiny-imagenet-200/train"
VAL_DIR="/home/neni/tiny-imagenet-200/val"

# BASELINE
# resnet50, no tf.data optimizations
#python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim --log_path ../logs/tensorflow/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler
# resnet50, no tf.data optimizations. only 3 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path ../logs/tensorflow/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --num_workers 3 --gpu_profiler
# resnet50, synthetic data
#python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path ../logs/tensorflow/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --synthetic_data
# resnet50, tf.data autotune optimizations
#python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path ../logs/tensorflow/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune

# DALI (GPU)
# resnet50, no tf.data optimizations
#python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali
# resnet50, no tf.data optimizations. only 3 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali
# resnet50, tf.data autotune optimizations 
# (which is basically setting 8 workers + autotune prefetch)
#python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --autotune --use_dali

# DALI (CPU)
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu
# resnet50, tf.data autotune optimizations
# (which is basically setting 8 workers + autotune prefetch)
python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --autotune --use_dali --dali_cpu