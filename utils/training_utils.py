import time
from contextlib import contextmanager
import cupy as cp

def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, 0, t
        start = time.time()


def timed_generator_cp(gen):
    start_gpu = cp.cuda.Event()
    end_gpu = cp.cuda.Event()
    start_cpu = time.time()

    start_gpu.record()
    for g in gen:
        end_cpu = time.time()
        end_gpu.record()
        end_gpu.synchronize()
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = end_cpu - start_cpu
        yield g, t_gpu/1000, t_cpu
        start_cpu = time.time()
        start_gpu.record()

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function

def timed_function_cp(f):
    def _timed_function(*args, **kwargs):
        start_gpu = cp.cuda.Event(block=True)
        end_gpu = cp.cuda.Event(block=True)
        start_cpu = time.time()
        start_gpu.record()
        
        ret = f(*args, **kwargs)

        end_cpu = time.time()
        end_gpu.record()
        end_gpu.synchronize()
        t_gpu = cp.cuda.get_elapsed_time(start_gpu, end_gpu)
        t_cpu = end_cpu - start_cpu

        return ret, t_gpu/1000, t_cpu
    return _timed_function

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result