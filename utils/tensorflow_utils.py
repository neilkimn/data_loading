import tensorflow as tf

def prefetched_loader(loader, device):

    for x, y in loader:
        with tf.device(device):
            input, target = x, y

        yield input, target