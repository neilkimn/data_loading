from math import ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import argparse
import tqdm
from utils.models import ResNet50_TF, SyntheticModelTF
from utils.data_generation import ImageNetDataTF, ImageNetDataDALI
from utils.training_utils import timed_function, timed_generator, timed_function_cp, timed_generator_cp
from utils.training_logging import BenchLogger, TimingProfiler, GPUProfiler
from utils.tensorflow_utils import prefetched_loader, limit_memory_growth

def parseargs():
    parser = argparse.ArgumentParser(usage="")
    parser.add_argument('--name')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('-bs','--batch_sizes', nargs='+', help='<Required> Set flag', required=True, type=int)
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--use_dali', action='store_true')
    parser.add_argument('--dali_cpu', action='store_true')
    parser.add_argument('--limit_memory_growth', action='store_true')
    parser.add_argument('-pf', '--prefetch', default=0, type=int)
    parser.add_argument('-df', '--default_optimizations', action='store_true')

    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--train_path', default="/home/neni/tiny-imagenet-200/train", type=str)
    parser.add_argument('--test_path', default="/home/neni/tiny-imagenet-200/val", type=str)
    parser.add_argument('--log_path', type=str)
    parser.add_argument('--gpu_profiler', action='store_true')

    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--height', default=64, type=int)
    parser.add_argument('--num_classes', default=200, type=int)
    parser.add_argument('--crop', default=64, type=int)
    args = parser.parse_args()
    return args

def get_loaders(batch_size, iterations, args):
    if args.use_dali:
        imagenet_dali = ImageNetDataDALI(batch_size, iterations, "tensorflow", args)
        train_loader = imagenet_dali.train_loader
        val_loader = imagenet_dali.val_loader
    else:
        imagenet_data = ImageNetDataTF(batch_size=batch_size, args=args)
        train_loader = imagenet_data.train_ds
        val_loader = imagenet_data.val_ds

    return train_loader, val_loader

if __name__ == '__main__':
    args = parseargs()
    channels = 3
    num_classes = args.num_classes
    #examples = 1_281_167
    examples = 100_000

    #epochs = 3
    epochs = 5


    if args.limit_memory_growth:
        limit_memory_growth()

    args.options = tf.data.Options()
    if args.autotune or args.default_optimizations:
        print("Apply default tf.data optimizations? True")
        args.options.experimental_optimization.apply_default_optimizations = True
    else:
        print("Apply default tf.data optimizations? False")
        args.options.experimental_optimization.apply_default_optimizations = False

    device = "GPU:0"

    experiment_name = args.name

    if args.autotune:
        experiment_name += "-AUTOTUNE"
    elif args.num_workers > 1:
        experiment_name += f"-{args.num_workers}-workers"
    
    if args.synthetic_data:
        experiment_name += "-synthetic_data"

    print("Experiment name: ", experiment_name)

    if args.log_path:
        timing_profiler = TimingProfiler(args.log_path, experiment_name)

    print("Evaluating batch sizes:", args.batch_sizes)

    for batch_size in args.batch_sizes:
        iterations = ceil(examples / batch_size)
        print(f"Total number of iterations: {iterations}, based on {examples} examples")
        logger_cls = BenchLogger("Train", batch_size, 0) # 0 is warmup iter
        if args.log_path:
            gpu_profiler_epoch = epochs-1
            gpu_profiler = GPUProfiler(args.log_path, experiment_name, batch_size, gpu_profiler_epoch)

        tf.keras.backend.clear_session()
        tf.keras.backend.set_image_data_format('channels_last')

        train_loader, val_loader = get_loaders(batch_size, iterations, args)

        model = ResNet50_TF(num_classes, args.crop, args.crop)

        step = timed_function_cp(model.train_step)

        for epoch in range(epochs):
            start = time.time()

            model.train_loss.reset_states()
            model.train_accuracy.reset_states()
            model.valid_loss.reset_states()
            model.valid_accuracy.reset_states()

            if epoch == gpu_profiler_epoch and args.log_path:
                gpu_profiler.start()
                gpu_profiler_started = time.time()

            #for i, ((images, labels), dt_gpu, dt_cpu) in enumerate(timed_generator_cp(train_loader)):
            for i, ((images, labels), dt_gpu, dt_cpu) in enumerate(timed_generator_cp(train_loader)):

                _, bt, _ = step(images, labels)

                logger_cls.iter_callback({"batch_time": bt, "data_time_cpu": dt_cpu, "data_time_gpu": dt_gpu})

                if epoch == gpu_profiler_epoch and args.log_path:
                    if ((time.time() - gpu_profiler_started) > 300) and gpu_profiler.running:
                        gpu_profiler.stop()
                        print("Stopped GPU profiler after 5 min")

                if i > iterations:
                    break

            end = time.time()

            batch_time, data_time_cpu, data_time_gpu = logger_cls.end_callback()

            timing_profiler.write_row(epoch, batch_size, end-start, batch_time, data_time_cpu, data_time_gpu)

            logger_cls.reset()

            if (epoch == gpu_profiler_epoch and args.log_path):
                gpu_profiler.stop()

            print(
                f"Epoch: [{epoch}/{epochs}]\t \
                  loss: {model.train_loss.result()}\t \
                  acc: {model.train_accuracy.result()}\
                  total time: {end - start}")
            
        if args.validation:
            for valid_images, valid_labels in val_loader:
                model.val_step(valid_images, valid_labels)
            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                    "valid loss: {:.5f}, valid accuracy: {:.5f}".format(
                        epoch + 1,
                        epochs,
                        model.train_loss.result(),
                        model.train_accuracy.result(),
                        model.valid_loss.result(),
                        model.valid_accuracy.result()))