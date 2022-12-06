import argparse
from math import ceil
import time

import torch
from torch import nn
from torch import optim
from utils.training_utils import nullcontext

from utils.data_generation import ImageNetDataTorch, ImageNetDataDALI
from utils.training_logging import BenchLogger, AverageMeter, TimingProfiler, GPUProfiler
from utils.training_utils import timed_function, timed_generator, timed_function_cp, timed_generator_cp, to_python_float
from utils.pytorch_utils import ModelAndLoss, get_train_step, get_val_step, get_prefetched_loader


def parseargs():
    parser = argparse.ArgumentParser(usage="")
    parser.add_argument('--name')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('-bs','--batch_sizes', nargs='+', help='<Required> Set flag', required=True, type=int)
    parser.add_argument('--batch_size', default=0, type=int)
    parser.add_argument('--use_dali', action='store_true')
    parser.add_argument('--dali_cpu', action='store_true')
    parser.add_argument('--output_type', type=str)
    parser.add_argument('--dlprof', action='store_true')

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
        if args.num_workers == 0:
            args.num_workers = 1
        if args.use_dali and args.synthetic_data and not args.output_type:
            raise ValueError("Need to set output type when using DALI + synthetic data")
        imagenet_dali = ImageNetDataDALI(batch_size, iterations, "pytorch", args)
        train_loader = imagenet_dali.train_loader
        val_loader = imagenet_dali.val_loader
    else:
        imagenet_torch = ImageNetDataTorch(batch_size, iterations, args)
        train_loader = imagenet_torch.train_ds
        val_loader = imagenet_torch.val_ds

    return train_loader, val_loader

if __name__ == '__main__':
    args = parseargs()
    channels = 3
    num_classes = args.num_classes
    examples = 100_000
    #train_loader_len = int(train_loader._size / args.batch_size)

    epochs = 3

    args.device = torch.device(f"cuda:0")

    # TODO: Add prefetch=True for tf.data autotune-comparable workload

    if args.num_workers <= 1:
        args.num_workers = 0

    experiment_name = args.name

    if args.num_workers:
        experiment_name += f"-{args.num_workers}-workers"
    elif args.synthetic_data:
        experiment_name += "-synthetic_data"

    print("Experiment Name:", experiment_name)

    if args.log_path:
        timing_profiler = TimingProfiler(args.log_path, experiment_name)

    if args.dlprof:
        import nvidia_dlprof_pytorch_nvtx
        nvidia_dlprof_pytorch_nvtx.init()

    for batch_size in args.batch_sizes:
        iterations = ceil(examples / batch_size)
        print(f"Total number of iterations: {iterations}, based on {examples} examples")
        logger_cls = BenchLogger("Train", batch_size, 0) # 0 is warmup iter
        train_top1 = AverageMeter()
        train_top5 = AverageMeter()
        train_loss = AverageMeter()

        if args.log_path:
            gpu_profiler_epoch = epochs-1
            gpu_profiler = GPUProfiler(args.log_path, experiment_name, batch_size, gpu_profiler_epoch)

        train_loader, val_loader = get_loaders(batch_size, iterations, args)

        loss_fn = nn.CrossEntropyLoss
        model_and_loss = ModelAndLoss("resnet50", loss_fn, args)
        optimizer = optim.SGD(model_and_loss.model.parameters(), lr=0.001, momentum=0.9)
        
        model_and_loss.model.train()

        step = get_train_step(model_and_loss, optimizer)
        step = timed_function_cp(step)
        prefetched_loader = get_prefetched_loader("dali" if args.use_dali else "pytorch")

        with torch.autograd.profiler.emit_nvtx() if args.dlprof else nullcontext():
            for epoch in range(epochs):
                start = time.time()

                if epoch == gpu_profiler_epoch and args.log_path:
                    gpu_profiler.start()
                    gpu_profiler_started = time.time()

                for i, ((images, labels), dt_gpu, dt_cpu) in enumerate(timed_generator_cp(prefetched_loader(train_loader, args.device))):
                    (loss, prec1, prec5), bt, _ = step(images, labels)
                    
                    prec1 = to_python_float(prec1)
                    prec5 = to_python_float(prec5)
                    loss = to_python_float(loss)

                    train_top1.update(prec1, images.size(0))
                    train_top5.update(prec5, images.size(0))
                    train_loss.update(loss, images.size(0))

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

                if epoch == gpu_profiler_epoch and args.log_path and args.gpu_profiler:
                    gpu_profiler.stop()

                print(
                    f"Epoch: [{epoch}/{epochs}] \
                    loss: {train_loss.avg} \
                    acc@1: {train_top1.avg} \
                    acc@5: {train_top5.avg} \
                    total time: {end - start}")
            
        if args.validation:
            step = get_val_step(model_and_loss)

            val_top1 = AverageMeter()
            val_top5 = AverageMeter()
            val_loss = AverageMeter()

            model_and_loss.model.eval()

            for input, target in prefetched_loader(val_loader, args.device):
                loss, prec1, prec5 = step(input, target)

                val_top1.update(to_python_float(prec1), input.size(0))
                val_top5.update(to_python_float(prec5), input.size(0))
                val_loss.update(to_python_float(loss), input.size(0))

            print(f"Validation\t \
                  loss: {val_loss.avg}\t \
                  acc@1: {val_top1.avg}\t \
                  acc@5: {val_top5.avg}")