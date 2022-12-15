#!/bin/bash

TRAIN_DIR="/home/neni/imagenet/train"
VAL_DIR="/home/neni/imagenet/val"

LOG_PATH="../logs/tensorflow/imagenet_200k"

# BASELINE
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_no_optim --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth
# resnet50, synthetic data
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --synthetic_data -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth
# resnet50, tf.data autotune optimizations
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth

LOG_PATH="../logs/tensorflow_dali_gpu/imagenet_200k"

# DALI (GPU)
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_no_optim_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --limit_memory_growth -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --limit_memory_growth -bs 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, no tf.data optimizations. 8 workers
#python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --limit_memory_growth -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000

# resnet50, no tf.data optimizations. Prefetch 1
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_GPU_prefetch_one --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali -pf 1 --limit_memory_growth -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000
# resnet50, tf.data autotune optimizations 
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali --limit_memory_growth -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000

LOG_PATH="../logs/tensorflow_dali_cpu/imagenet_200k"

# DALI (CPU)
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_no_optim_CPU_GETOLDLOG --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth
# resnet50, no tf.data optimizations. 8 workers
#python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000

# resnet50, tf.data autotune optimizations (+ NO DALI workers)
# located at old_dgx_logs/tensorflow_dali_cpu/imagenet_200k/profiler-tensorflow_resnet50_CPU-AUTOTUNE.csv

# resnet50, tf.data autotune optimizations (+ 3 DALI workers)
python ../src/train_tensorflow.py --name SMI_tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --autotune --use_dali --dali_cpu -bs 64 128 256 --width 256 --height 256 --crop 224 --num_classes 1000 --limit_memory_growth



