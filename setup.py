"""Installation script for the 'timechamber' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os

root_dir = os.path.dirname(os.path.realpath(__file__))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym",
    "torch",
    "omegaconf",
    "termcolor",
    "dill",
    "hydra-core>=1.1",
    "rl-games==1.5.2",
    "pyvirtualdisplay",
    "multielo @ git+https://github.com/djcunningham0/multielo.git@440f7922b90ff87009f8283d6491eb0f704e6624",
    "matplotlib==3.5.2",
    "pytest==7.1.2",
]

# Installation operation
setup(
    name="timechamber",
    author="ZeldaHuang",
    version="0.0.1",
    description="Super fast self-play framework via parallel techniques",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7, 3.8"],
    zip_safe=False,
)

# EOF
