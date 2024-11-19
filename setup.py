#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="ddi_kt_2024",
    version="0.0.1",
    description="No des",
    author="",
    author_email="",
    url="https://github.com/user/project",
    install_requires=[],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)