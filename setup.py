# MIT License
#
# Copyright (c) 2022 Quandela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# As a special exception, the copyright holders of exqalibur library give you
# permission to combine exqalibur with code included in the standard release of
# Perceval under the MIT license (or modified versions of such code). You may
# copy and distribute such a combined system following the terms of the MIT
# license for both exqalibur and Perceval. This exception for the usage of
# exqalibur is limited to the python bindings used by Perceval.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import os
from setuptools import setup

with open("pennylane_perceval/_version.py") as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

# Put pip installation requirements here.
# Requirements should be as minimal as possible.
# Avoid pinning, and use minimum version numbers
# only where required.
requirements = ["pennylane>=0.37.0", "perceval-quandela>=0.10.5"]


info = {
    # 'name' is the name that will be used by pip for installation
    "name": "PennyLane-Perceval",
    "version": version,
    "maintainer": "Quandela",
    "url": "https://www.quandela.com/",
    "license": "MIT License",
    "packages": [
        # The name of the folder containing the plugin.
        # This is the name that will be used when importing
        # the plugin in Python.
        "pennylane_perceval"
    ],
    "entry_points": {
        "pennylane.plugins": [
            # List the short name of each device provided by
            # the plugin, as well as the path to the Device class
            # it corresponds to in the plugin. This allows
            # the device to be imported automatically via the
            # `pennylane.device` device loader.
            "perceval.dev = pennylane_perceval:PercevalDevice",
        ]
    },
    # Place a one line description here. This will be shown by pip
    "description": "PennyLane plugin to access the Quandela cloud service.",
    "long_description": open("README.rst").read(),
    # The name of the folder containing the plugin
    "provides": ["pennylane_perceval"],
    "install_requires": requirements,
}

classifiers = [
    "Development Status :: 4 - Beta",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: POSIX",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Microsoft :: Windows",
    "Programming Language :: Python",
    # Make sure to specify here the versions of Python supported
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3 :: Only",
    "Topic :: Scientific/Engineering :: Physics",
]

setup(classifiers=classifiers, **(info))
