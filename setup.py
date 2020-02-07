#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

"""

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Package meta-data.
NAME = 'torch-sublstm'
DESCRIPTION = 'A subLSTM implementation as described in the article \
    "Cortical microcircuits as gated-recurrent neural networks" using PyTorch'
URL = 'https://github.com/mllera14/subLSTM'
EMAIL = 'm.lleramontero@bristol.ac.uk'
AUTHOR = 'Milton Llera Montero'
REQUIRES_PYTHON  = '>= 3.6.0'
VERSION = '0.1.0'

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy',
    'torch'
]

PACKAGES = [
    'subLSTM'
]

PACKAGE_DIR = {'' : 'src'}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.rst' is present in your MANIFEST.in file!
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, 'src', '__version__.py')) as f:
    exec(f.read(), about)


class UploadCommand(Command):
    #Support setup.py upload

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        # Prints things in bold.
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system('{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPi via Twine…')
        os.system('twine upload dist/*')

        sys.exit()


# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=PACKAGES,
    package_dir = PACKAGE_DIR,
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers

        'Development Status :: 1 - Alpha',

        'Intended Audience :: Science/Research'
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Computational Neuroscience',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    ext_modules=[
        CUDAExtension('sublstm_cuda', [
            os.path.join(here, 'src', 'subLSTM', 'basic', 'sublstm.cpp'),
            os.path.join(here, 'src', 'subLSTM', 'basic', 'sublstm.cu')
        ]),
    ],
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'build_ext': BuildExtension
    },
)
"""

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGES = [
    'subLSTM'
]

PACKAGE_DIR = {'' : 'src'}

setup(
    name='torch-sublstm',
    packages=PACKAGES,
    package_dir=PACKAGE_DIR,
    ext_modules=[
        CUDAExtension('sublstm_cuda', [
            'sublstm.cpp',
            'sublstm.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })