#!/bin/bash

TRAIN_DIR="/home/neni/tiny-imagenet-200/train"
VAL_DIR="/home/neni/tiny-imagenet-200/val"

LOG_PATH="../logs/tensorflow/tiny_imagenet64"

# BASELINE
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler -bs 128 256 512
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 -bs 128 256 512
# resnet50, synthetic data
python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --synthetic_data -bs 128 256 512
# resnet50, tf.data autotune optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50 --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune -bs 128 256 512

LOG_PATH="../logs/tensorflow_dali_gpu/tiny_imagenet64"

# DALI (GPU)
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --limit_memory_growth -bs 128 256 512
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --limit_memory_growth -bs 128 256 512
# resnet50, no tf.data optimizations. 8 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --limit_memory_growth -bs 128 256 512

# resnet50, no tf.data optimizations. Prefetch 1
python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU_prefetch_one --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali -pf 1 --limit_memory_growth -bs 128 256 512
# resnet50, tf.data autotune optimizations 
python ../src/train_tensorflow.py --name tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali -bs 128 256 512 --limit_memory_growth

# tf.data autotune optimizations + DALI GPU where tensorflow should hog all GPU memory. Should see contention for GPU memory?
python ../src/train_tensorflow.py --name NO_MEM_LIMIT_AUTOTUNE_tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali -bs 128 256 512
# tf.data autotune optimizations + DALI GPU where GPU memory growth is limited. Should be the correct case
python ../src/train_tensorflow.py --name MEM_LIMIT_AUTOTUNE_tensorflow_resnet50_GPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali --limit_memory_growth -bs 128 256 512

LOG_PATH="../logs/tensorflow_dali_cpu/tiny_imagenet64"

# DALI (CPU)
# resnet50, no tf.data optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --use_dali --dali_cpu -bs 128 256 512 --limit_memory_growth
# resnet50, no tf.data optimizations. only 3 workers
python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 3 --use_dali --dali_cpu -bs 128 256 512 --limit_memory_growth
# resnet50, no tf.data optimizations. 8 workers
#python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --num_workers 8 --use_dali --dali_cpu -bs 128 256 512
# resnet50, tf.data autotune optimizations
python ../src/train_tensorflow.py --name tensorflow_resnet50_CPU --log_path $LOG_PATH --train_path $TRAIN_DIR --test_path $VAL_DIR --gpu_profiler --autotune --use_dali --dali_cpu -bs 128 256 512 --limit_memory_growth