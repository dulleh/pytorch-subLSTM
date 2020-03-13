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
    parser = argparse.ArgumentParser(description='Addition task')
    path_to_this = os.path.abspath(os.path.dirname(__file__))
    py_file_name = 'results/subLSTM_1_350/trace.csv'
    cuda_file_name = 'results/subLSTMCuda_1_350/trace.csv'
    py_save_path = os.path.join(path_to_this, py_file_name)
    cuda_save_path = os.path.join(path_to_this, cuda_file_name)

    pytimes = np.loadtxt(py_save_path, delimiter=',')
    cudatimes = np.loadtxt(cuda_save_path, delimiter=',')
    plt.suptitle('Comparison of subLSTM Training Loss Between Implementations')
    plt.plot(np.arange(1, len(pytimes)+1, 1), pytimes, label='subLSTM Py', color='tab:orange', linewidth=2.0)
    plt.plot(np.arange(1, len(cudatimes)+1, 1), cudatimes, label='subLSTM CUDA', color='tab:green', linestyle='dotted', linewidth=3.0)
    plt.legend()
    plt.xlabel('Batches Trained On / 10')
    plt.ylabel('Error')
    plt.show()
