

from math import ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import argparse
import tqdm
#from profiler import TimingProfiler
from utils.models import ResNet50_TF, SyntheticModelTF
from utils.data_generation import ImageNetDataTF

def parseargs():
    parser = argparse.ArgumentParser(usage='Test GPU transfer speed in TensorFlow(default) and Pytorch.')
    parser.add_argument('--name')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--to_gpu', action='store_true')
    parser.add_argument('--train_path', default="tiny-imagenet-200/train", type=str)
    parser.add_argument('--test_path', default="tiny-imagenet-200/val/images", type=str)
    parser.add_argument('--crop', default=64, type=int)
    args = parser.parse_args()
    return args

def run_experiments(run, batch_size, iterations, height, width, num_classes, args):
    
    model = ResNet50_TF(num_classes, 64, 64)

    imagenet_data = ImageNetDataTF(img_height=height, img_width=width, batch_size=batch_size, args=args)
    train_loader = iter(imagenet_data.train_ds)
    start = time.time()

    for step in tqdm.tqdm(range(iterations)):
        (images, labels) = next(train_loader)

        with tf.device("GPU:0"):
            _ = model.train_step(images, labels)

    end = (time.time() - start)
    print(f"1 Epoch train: {end}s, batch size: {batch_size}")
    print("\n")

    #timing_profiler.write_row(run, batch_size, args.synthetic_data, args.to_gpu, args.preprocessing, end)

if __name__ == '__main__':
    args = parseargs()
    width = 64
    height = 64
    channels = 3
    num_classes = 200

    if args.autotune:
        experiment_name = f"{args.name}-AUTOTUNE"
    elif args.num_workers:
        experiment_name = f"{args.name}-{args.num_workers}-workers"
    elif args.synthetic_data:
        experiment_name = f"{args.name}-synthetic_data"

    print("Experiment name: ", experiment_name)

    runs = 5

    for batch_size in [128]:
        iterations = ceil(100_000 / batch_size)

        for run in range(1, runs+1):
            tf.keras.backend.clear_session()
            run_experiments(run, batch_size, iterations, height, width, num_classes, args) # no options
            os.system('sudo sh -c "/bin/echo 3 > /proc/sys/vm/drop_caches"')