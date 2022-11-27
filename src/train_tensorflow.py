from math import ceil
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
import argparse
import tqdm
from utils.models import ResNet50_TF, SyntheticModelTF
from utils.data_generation import ImageNetDataTF, ImageNetDataDALI
from utils.training_utils import timed_function, timed_generator
from utils.training_logging import BenchLogger, AverageMeter
from utils.tensorflow_utils import prefetched_loader

def parseargs():
    parser = argparse.ArgumentParser(usage='Test GPU transfer speed in TensorFlow(default) and Pytorch.')
    parser.add_argument('--name')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--batch_size', default=0, type=int)
    parser.add_argument('--autotune', action='store_true')
    parser.add_argument('--use_dali', action='store_true')
    parser.add_argument('--dali_cpu', action='store_true')

    parser.add_argument('--synthetic_data', action='store_true')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--train_path', default="/home/neni/tiny-imagenet-200/train", type=str)
    parser.add_argument('--test_path', default="/home/neni/tiny-imagenet-200/val", type=str)

    parser.add_argument('--width', default=64, type=int)
    parser.add_argument('--height', default=64, type=int)
    parser.add_argument('--num_classes', default=200, type=int)
    parser.add_argument('--crop', default=64, type=int)
    args = parser.parse_args()
    return args

def get_loaders(batch_size, iterations, args):
    if args.use_dali:
        imagenet_dali = ImageNetDataDALI(height, width, batch_size, iterations, "tensorflow", args)
        train_loader = imagenet_dali.train_loader
        val_loader = imagenet_dali.val_loader
    else:
        imagenet_data = ImageNetDataTF(img_height=height, img_width=width, batch_size=batch_size, args=args)
        train_loader = imagenet_data.train_ds
        val_loader = imagenet_data.val_ds

    return train_loader, val_loader

if __name__ == '__main__':
    args = parseargs()
    width = args.width
    height = args.height
    channels = 3
    num_classes = args.num_classes

    epochs = 5

    device = "GPU:0"

    if args.autotune:
        experiment_name = f"{args.name}-AUTOTUNE"
    elif args.num_workers:
        experiment_name = f"{args.name}-{args.num_workers}-workers"
    elif args.synthetic_data:
        experiment_name = f"{args.name}-synthetic_data"

    if args.use_dali and args.num_workers == 0:
        raise ValueError(f"Cannot use DALI with no workers and tf.AUTOTUNE: {args.autotune}")

    print("Experiment name: ", experiment_name)

    for batch_size in [args.batch_size]:
        iterations = ceil(100_000 / batch_size)
        logger_cls = BenchLogger("Train", batch_size, 0) # 0 is warmup iter

        tf.keras.backend.clear_session()
        tf.keras.backend.set_image_data_format('channels_last')

        train_loader, val_loader = get_loaders(batch_size, iterations, args)

        model = ResNet50_TF(num_classes, height, width)

        step = timed_function(model.train_step)

        for epoch in range(epochs):
            model.train_loss.reset_states()
            model.train_accuracy.reset_states()
            model.valid_loss.reset_states()
            model.valid_accuracy.reset_states()

            for i, ((images, labels), dt) in enumerate(timed_generator(train_loader)):

                _, bt = step(images, labels)

                logger_cls.iter_callback({"batch_time": bt, "data_time": dt})
                    
            total_img_s, compute_img_s, prep_img_s, batch_time, data_time = logger_cls.end_callback()

            logger_cls.reset()

            print(
                f"Epoch: [{epoch}/{epochs}]\t \
                  loss: {model.train_loss.result()}\t \
                  acc: {model.train_accuracy.result()}")
            
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