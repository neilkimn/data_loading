import tensorflow as tf
import torch
from torchvision import models

### ACTUAL MODELS

class ResNet50_TF():
    def __init__(self, num_classes, height, width):
        inputs = tf.keras.Input(shape=(None, None, 3))
        outputs = tf.keras.applications.ResNet50(
            weights=None, input_shape=(height, width, 3), classes=num_classes
        )(inputs)
        self.model = tf.keras.Model(inputs, outputs)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
        self.model.compile(
            optimizer=self.optimizer, 
            loss=self.loss_fn, 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
        )
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

        self.train_loss(loss_value)
        self.train_accuracy(y, logits)
        return loss_value

    @tf.function
    def val_step(self, x, y):
        predictions = self.model(x, training=False)
        v_loss = self.loss_fn(y, predictions)

        self.valid_loss(v_loss)
        self.valid_accuracy(y, predictions)

class ResNet_PT(torch.nn.Module):
    def __init__(self, name, num_classes):
        super(ResNet_PT, self).__init__()
        model = models.__dict__[name](num_classes=num_classes)
        model.cuda()

        self.model = model

    def forward(self, x):
        return self.model(x)

### SYNTHETIC MODELS

class SyntheticModelTF():
    def __init__(self, to_gpu=True) -> None:
        self.to_gpu = to_gpu

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        self.valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(self, x, y):
        if self.to_gpu:
            return tf.cast(x, dtype=tf.float32)[0, 0], tf.cast(y, dtype=tf.float32)[0]
        with tf.device('/CPU:0'):
            return tf.cast(x, dtype=tf.float32)[0, 0], tf.cast(y, dtype=tf.float32)[0]

class SyntheticModelPT(torch.nn.Module):
    def __init__(self, to_gpu):
        super(SyntheticModelPT, self).__init__()
        self.to_gpu = to_gpu

    def forward(self, x, y):
        with torch.no_grad():
            if self.to_gpu:
                return x.cuda()[0, 0], \
                       y.cuda()[0]
            return x.cpu(), \
                   y.cpu()