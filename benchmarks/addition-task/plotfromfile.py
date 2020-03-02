# coding: utf-8

import sys
import os
import argparse
import time
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, '../../src/')
sys.path.insert(0, '../')

from wrappers import init_model
from utils import train, test, drawepochs

import matplotlib.pyplot as plt

if __name__ == '__main__':
    batchsize, seqlen, trainingsize, numepochs = 4, 4, 80, 4
    parser = argparse.ArgumentParser(description='Addition task')
    path_to_this = os.path.abspath(os.path.dirname(__file__))
    cuda_file_name = 'CUDA_v1_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    python_file_name = 'python_v1_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    cuda_save_path = os.path.join(path_to_this, cuda_file_name)
    python_save_path = os.path.join(path_to_this, python_file_name)

    cudatimes = np.loadtxt(cuda_save_path, delimiter=',')
    pythontimes = np.loadtxt(python_save_path, delimiter=',')
    plt.suptitle('Avg. Forward Time per Epoch Vs Hidden Units with Batch Size {}, Seq. Length {}, Training Size {} across {} epochs'.format(batchsize, seqlen, trainingsize, numepochs))
    plt.plot(np.arange(1, len(pythontimes)+1, 1), pythontimes, label='Python')
    plt.plot(np.arange(1, len(pythontimes)+1, 1), cudatimes, label='CUDA forward, C++ backward')
    plt.legend()
    plt.xlabel('Hidden units')
    plt.ylabel('Average Time (s)')
    plt.show()
