#!/bin/bash

TRAIN_DIR="/home/neni/tiny-imagenet-200/train"
VAL_DIR="/home/neni/tiny-imagenet-200/val"

# TODO: Try to change tf.data autotune with DALI to use 
# `options.experimental_optimization.apply_default_optimizations = True` and apply to DALIDataset

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
# resnet50, no tf.data optimizations. 8 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali
# resnet50, tf.data autotune optimizations 
#python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali

# tf.data autotune optimizations + DALI GPU where tensorflow should hog all GPU memory. Should see contention for GPU memory?
#python ../src/train_tensorflow.py --name NO_MEM_LIMIT_AUTOTUNE_tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali
# tf.data autotune optimizations + DALI GPU where GPU memory growth is limited. Should be the correct case
#python ../src/train_tensorflow.py --name MEM_LIMIT_AUTOTUNE_tensorflow_resnet50_GPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali --limit_memory_growth

# DALI (CPU)
# resnet50, no tf.data optimizations
#python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu
# resnet50, no tf.data optimizations. only 3 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu
# resnet50, no tf.data optimizations. 8 workers
python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --dali_cpu
# resnet50, tf.data autotune optimizations
#python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path ../logs/tensorflow_dali/imagenet64/ --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali --dali_cpu