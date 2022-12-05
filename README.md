### Prerequisites

For python3.6: `curl https://bootstrap.pypa.io/pip/3.6/get-pip.py | python -`

Else: `curl https://bootstrap.pypa.io/get-pip.py | python -`

```shell
$ pip install -U setuptools

$ pip install -e .

$ pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110

$ pip install nvidia-dlprof[pytorch]

$ pip install nvidia-dlprofviewer
```

### Run experiments

```shell
python ../src/train_tensorflow.py --name tensorflow_resnet50_no_optim --log_path ../logs/tensorflow/imagenet64/ --train_path /home/neni/repos/thesis/tiny-imagenet-200/train --test_path /home/neni/repos/thesis/tiny-imagenet-200/val --gpu_profiler

python ../src/train_tensorflow.py --name tensorflow_resnet50_synthetic_data --log_path ../logs/tensorflow/imagenet64/ --train_path /home/neni/repos/thesis/tiny-imagenet-200/train --test_path /home/neni/repos/thesis/tiny-imagenet-200/val --gpu_profiler --synthetic_data

python ../src/train_tensorflow.py --name tensorflow_resnet50_AUTOTUNE --log_path ../logs/tensorflow/imagenet64/ --train_path /home/neni/repos/thesis/tiny-imagenet-200/train --test_path /home/neni/repos/thesis/tiny-imagenet-200/val --gpu_profiler --autotune
$ CUDA_VISIBLE_DEVICES=0 ./run_tensorflow.sh

$ CUDA_VISIBLE_DEVICES=0 ./run_pytorch.sh
```

### DLProf profiling

- Example

```shell
$ dlprof --mode pytorch --reports summary --duration 120 python ../src/train_pytorch.py --name pytorch_resnet50_no_optim --log_path ../logs/pytorch/imagenet64/ --train_path /home/neni/tiny-imagenet-200/train/ --test_path /home/neni/tiny-imagenet-200/val/ --dlprof
```

and then

```shell
$ dlprofviewer --bind 0.0.0.0 --port 45555 dlprof_dldb.sqlite
```

and then on host-side

```shell
ssh -v -N -L 45555:127.0.0.1:45555 user@remote
```
