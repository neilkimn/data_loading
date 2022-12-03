import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader as TorchDL, Dataset
import tensorflow as tf
from contextlib import redirect_stdout

from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
from nvidia.dali.plugin.tf import DALIDataset, DALIIterator
from nvidia.dali import Pipeline, pipeline_def, fn, types


### PYTORCH - DATA GENERATION + PREPROCESS ###

class ImageNetDataTorch():
    def __init__(self, img_height, img_width, batch_size, iterations, args):
        self.img_height = img_height
        self.img_width = img_width
        self.crop = args.crop
        self.batch_size = batch_size
        self.iterations = iterations
        self.num_workers = args.num_workers
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(3,1,1)

        if args.synthetic_data:
            train_ds = Infinite(self.iterations, self.batch_size, self.img_height, self.img_width, args.num_classes, args.device)

            print("Remember to double check validation data iterations!")
            val_iterations = 10_000 // batch_size
            val_ds = Infinite(val_iterations, self.batch_size, self.img_height, self.img_width, args.num_classes, args.device)

            self.num_workers = 0
        else:
            train_ds = ImageFolder(root=args.train_path)
            val_ds = ImageFolder(root=args.test_path)
        

            train_ds.transform = transforms.Compose([
                transforms.Resize(256), # Resize square
                transforms.RandomResizedCrop(self.crop), # Resize + Crop
                transforms.RandomHorizontalFlip(), # Horizontal Flip
                transforms.ToTensor(), # Normalize 1/2
                transforms.Normalize(self.mean, self.std) # Normalize 2/2
            ])

            val_ds.transform = transforms.Compose([transforms.Resize(self.crop), transforms.ToTensor()])

        self.train_ds = TorchDL(train_ds, batch_size=self.batch_size,
                                    shuffle=True, num_workers=self.num_workers)
        self.val_ds = TorchDL(train_ds, batch_size=self.batch_size,
                                shuffle=False, num_workers=self.num_workers)

### PYTORCH - SYNTHETIC INPUT ###

class Infinite(Dataset):
    def __init__(self, iterations, batch_size, img_height, img_width, num_classes, device):
        self.iterations = iterations
        self.batch_size = batch_size
        self.img = torch.randn(3, img_height, img_width).to(device)
        self.label = torch.randint(0, num_classes, (1,), dtype=torch.long).squeeze().to(device)

    @property
    def _size(self):
        return self.iterations*self.batch_size

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        return self.__getitem__(None)

### TENSORFLOW - DATA GENERATION ###

class ImageNetDataTF:
    def __init__(self, img_height, img_width, batch_size, args):
        if args.synthetic_data:
            self.train_ds = get_synth_input_fn(img_height, img_width, 3, args.num_classes, batch_size)
            self.val_ds = get_synth_input_fn(img_height, img_width, 3, args.num_classes, batch_size)
        else:
            train_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                args.train_path,
                seed=42,
                image_size=(img_height, img_width),
                batch_size=batch_size)
            train_ds.with_options(args.options)
            
            val_ds: tf.data.Dataset = tf.keras.utils.image_dataset_from_directory(
                args.test_path,
                seed=42,
                image_size=(img_height, img_width),
                batch_size=batch_size)
            if args.autotune:
                train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
                val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

            self.train_preprocessor = PreprocessingTF(img_height, args.crop)
            if args.autotune:
                self.train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (self.train_preprocessor(x), y), num_parallel_calls=tf.data.AUTOTUNE)
            elif args.num_workers:
                print(f"Setting {args.num_workers} parallel calls")
                self.train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (self.train_preprocessor(x), y), num_parallel_calls=args.num_workers)
            else:
                self.train_ds: tf.data.Dataset = train_ds.map(lambda x, y: (self.train_preprocessor(x), y))

            self.val_preprocessor = PreprocessingTF(img_height, args.crop)
            self.val_ds = val_ds.map(lambda x, y: (self.val_preprocessor(x, True), y))

### TENSORFLOW - PREPROCESS ###

@tf.function(jit_compile=False)
def normalize_img(image):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255.

class ToTensor(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return normalize_img(x)
    
class PreprocessingTF(tf.keras.layers.Layer):
    def __init__(self, image_size, image_crop):
        super(PreprocessingTF, self).__init__()
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

        self.preprocessing_layer = tf.keras.Sequential([
            tf.keras.layers.Resizing(image_size, image_size),
            tf.keras.layers.RandomCrop(image_crop, image_crop),
            tf.keras.layers.RandomFlip(mode="horizontal"),
            ToTensor(),
            tf.keras.layers.Normalization(mean=mean, variance=std)
        ])

        self.validation_layer = tf.keras.Sequential([
            tf.keras.layers.Resizing(image_crop, image_crop),
            ToTensor()
        ])
    
    @tf.function()
    def call(self, img, validation=False):
        if validation:
            return self.validation_layer(img)
        else:
            return self.preprocessing_layer(img)

### TENSORFLOW - SYNTHETIC INPUT ###

def get_synth_data(height, width, num_channels, num_classes):
    inputs = tf.random.truncated_normal([height, width, num_channels],
                                        dtype=tf.float32,
                                        mean=127,
                                        stddev=60,
                                        name='synthetic_inputs')
    labels = tf.random.uniform([1],
                                minval=0,
                                maxval=num_classes - 1,
                                dtype=tf.int32,
                                name='synthetic_labels')
    return inputs, labels

def get_synth_input_fn(height, width, num_channels, num_classes, batch_size, drop_remainder = True):
    def input_fn():
        inputs, labels = get_synth_data(
            height=height,
            width=width,
            num_channels=num_channels,
            num_classes=num_classes)
        # Cast to float32 for Keras model.
        labels = tf.cast(labels, dtype=tf.float32)
        data = tf.data.Dataset.from_tensors((inputs, labels)).repeat()

        # `drop_remainder` will make dataset produce outputs with known shapes.
        data = data.batch(batch_size, drop_remainder=drop_remainder)
        data = data.apply(tf.data.experimental.copy_to_device('/gpu:0'))
        data = data.cache()
        data = data.prefetch(tf.data.AUTOTUNE)
        return data

    return input_fn()


### DALI - DATA GENERATION ###

class ImageNetDataDALI:
    def __init__(self, img_height, img_width, batch_size, iterations, output_type, args):
        self.img_height = img_height
        self.img_width = img_width
        self.crop = args.crop
        self.batch_size = batch_size
        self.iterations = iterations
        self.output_type = output_type

        if args.synthetic_data:
            pipe_train = create_synthetic_pipeline(self.img_height, self.img_width, self.batch_size, self.iterations, args.num_classes, args.num_workers, args.output_type)

            print("Remember to double check validation data iterations!")
            val_iterations = 10_000 // self.batch_size

            pipe_val = create_synthetic_pipeline(self.img_height, self.img_width, self.batch_size, val_iterations, args.num_classes, args.num_workers, args.output_type)
        else:
            pipe_train = create_pipeline(
                batch_size=self.batch_size,
                num_threads=args.num_workers,
                data_dir=args.train_path,
                crop=(self.img_height, self.img_width),
                output_type=self.output_type,
                dali_cpu=args.dali_cpu
            )
            pipe_val = create_pipeline(
                batch_size=self.batch_size,
                num_threads=args.num_workers,
                data_dir=args.test_path,
                crop=(self.img_height, self.img_width),
                output_type=self.output_type,
                dali_cpu=args.dali_cpu
            )
        pipe_train.build()
        pipe_val.build()

        if self.output_type == "pytorch":
            self.train_loader = DALIGenericIterator(
                pipe_train,
                ["data", "label"],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )
            self.val_loader = DALIGenericIterator(
                pipe_val,
                ["data", "label"],
                reader_name="Reader",
                last_batch_policy=LastBatchPolicy.PARTIAL,
            )

        if self.output_type == "tensorflow":
            self.train_loader = DALIDataset(
                pipe_train,
                output_dtypes = (tf.float32, tf.int32),
                batch_size=self.batch_size,
                num_threads=args.num_workers
            )

            self.train_loader = self.train_loader.with_options(args.options)

            self.val_loader = DALIDataset(
                pipe_val,
                output_dtypes = (tf.float32, tf.int32),
                batch_size = self.batch_size,
                num_threads=args.num_workers
            )
            if args.autotune:
                self.train_loader = self.train_loader.prefetch(tf.data.AUTOTUNE)
                self.val_loader = self.val_loader.prefetch(tf.data.AUTOTUNE)
            else:
                self.train_loader = self.train_loader.prefetch(args.prefetch)
                self.val_loader = self.val_loader.prefetch(args.prefetch)

def create_pipeline(batch_size, num_threads, data_dir, crop, output_type, dali_cpu):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads)
    def _create_pipeline(data_dir, crop, output_type, dali_cpu=False):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            pad_last_batch=True,
            name="Reader",
        )

        dali_device = "cpu" if dali_cpu else "gpu"
        decoder_device = "cpu" if dali_cpu else "mixed"
        output_layout = "HWC" if output_type == "tensorflow" else "CHW"
        print("decoder device:", decoder_device)
        print("output layout:", output_layout)
        
        images = fn.decoders.image(inputs, device = decoder_device)

        images = fn.resize( # Resize square
            images,
            dtype=types.FLOAT,
            size=[256, 256]
        )

        images = fn.crop_mirror_normalize( # Resize + Crop and convert image type to proper dtype and layout
            images,
            dtype=types.FLOAT,
            output_layout=output_layout,
            crop=crop,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
            device=dali_device
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal = flip_coin, device = dali_device) # Horizontal Flip

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir, crop, output_type, dali_cpu)

def create_synthetic_pipeline(img_height, img_width, batch_size, iterations, num_classes, num_workers, output_type):
    iterator = InfiniteDALI(img_height, img_width, batch_size, iterations, num_classes, output_type)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_workers, device_id=0)
    with pipe:
        if output_type == "pytorch":
            dtypes = [types.FLOAT, types.INT64]
        if output_type == "tensorflow":
            dtypes = [types.FLOAT, types.INT32]
        images, labels = fn.external_source(source=iterator, num_outputs=2, device="gpu", dtype=dtypes)
        pipe.set_outputs(images, labels)

    return pipe


### DALI - SYNTHETIC INPUT ###
        
class InfiniteDALI(object):
    def __init__(self, iterations, batch_size, img_height, img_width, num_classes, output_type):
        self.iterations = iterations
        self.batch_size = batch_size
        if output_type == "pytorch":
            self.img = torch.randn(3, img_height, img_width)
            self.label = torch.randint(0, num_classes, (1,), dtype=torch.long).squeeze()
        if output_type == "tensorflow":
            self.img, self.label = get_synth_data(img_height, img_width, 3, num_classes)

    @property
    def _size(self):
        return self.iterations*self.batch_size

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return self._size

    def __iter__(self):
        return self

    def __next__(self):
        return self.__getitem__(None)