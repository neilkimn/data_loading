import tensorflow as tf

def prefetched_loader(loader, device):

    for x, y in loader:
        with tf.device(device):
            input, target = x, y

        yield input, target

# Set TensorFlow to not map all GPU memory visible to current process
# https://stackoverflow.com/questions/70782399/tensorflow-is-it-normal-that-my-gpu-is-using-all-its-memory-but-is-not-under-fu
# https://www.tensorflow.org/guide/gpu#limiting_gpu_memory_growth
def limit_memory_growth():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)