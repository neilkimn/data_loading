"""Setup script for package."""
import os
import sys
from typing import Optional

from setuptools import Command, find_packages, setup

setup(
    name="data_loading",
    version="0.0.1",
    description="Thesis setup and requirements.",
    author="Neil Kim Nielsen",
    url="",
    keywords="",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "torch",
        "torchvision",
        "torch_tb_profiler",
        "tensorboard",
        "tensorboard-plugin-profile",
        "tensorflow",
        "tqdm",
        "nvidia-dali-cuda110",
        "nvidia-dali-tf-plugin-cuda110"
    ],
    dependency_links=[
        "https://developer.download.nvidia.com/compute/redist"
    ],
    extras_require={
    },
    setup_requires=[
    ],
    classifiers=["Programming Language :: Python :: 3.9"],
    entry_points={
    },
    cmdclass={
    },
)
