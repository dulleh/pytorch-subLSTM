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

if __name__ == '__main__':
    path_to_this = os.path.abspath(os.path.dirname(__file__))
    cuda_file_name = 'backwardtimes.csv'
    cuda_save_path = os.path.join(path_to_this, cuda_file_name)
    cudatimes = np.loadtxt(cuda_save_path, delimiter=',')
    total = np.sum(cudatimes)
    print(total)
