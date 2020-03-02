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
    cuda_v1_file_name = 'CUDA_v1_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    cuda_fused_file_name = 'CUDA_fused_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    python_v1_file_name = 'python_v1_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    python_fused_file_name = 'python_fused_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    lstm_unfused_file_name = 'LSTM_unfused_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    lstm_fused_file_name = 'LSTM_fused_batch{}_seq{}_train{}_epochs{}.csv'.format(batchsize, seqlen, trainingsize, numepochs)
    cuda_v1_save_path = os.path.join(path_to_this, cuda_v1_file_name)
    cuda_fused_save_path = os.path.join(path_to_this, cuda_fused_file_name)
    python_v1_save_path = os.path.join(path_to_this, python_v1_file_name)
    python_fused_save_path = os.path.join(path_to_this, python_fused_file_name)
    lstm_unfused_save_path = os.path.join(path_to_this, lstm_unfused_file_name)
    lstm_fused_save_path = os.path.join(path_to_this, lstm_fused_file_name)

    cudav1times = np.loadtxt(cuda_v1_save_path, delimiter=',')
    cudafusedtimes = np.loadtxt(cuda_fused_save_path, delimiter=',')
    pythonv1times = np.loadtxt(python_v1_save_path, delimiter=',')
    pythonfusedtimes = np.loadtxt(python_fused_save_path, delimiter=',')
    lstmunfusedtimes = np.loadtxt(lstm_unfused_save_path, delimiter=',')
    lstmfusedtimes = np.loadtxt(lstm_fused_save_path, delimiter=',')
    plt.suptitle('Avg. Forward Time per Epoch Vs Hidden Units with Batch Size {}, Seq. Length {}, Training Size {} across {} epochs'.format(batchsize, seqlen, trainingsize, numepochs))
    plt.plot(np.arange(1, len(pythonv1times)+1, 1), pythonv1times, label='Python (run alongside unfused)')
    plt.plot(np.arange(1, len(pythonfusedtimes)+1, 1), pythonfusedtimes, label='Python (run alongside fused)')
    plt.plot(np.arange(1, len(cudav1times)+1, 1), cudav1times, label='CUDA forward (unfused, FML)')
    plt.plot(np.arange(1, len(cudafusedtimes)+1, 1), cudafusedtimes, label='CUDA forward (fused, FML)')
    plt.plot(np.arange(1, len(lstmunfusedtimes)+1, 1), lstmunfusedtimes, label='LSTM unfused')
    plt.plot(np.arange(1, len(lstmfusedtimes)+1, 1), lstmfusedtimes, label='LSTM (pytorch standard)')
    plt.legend()
    plt.xlabel('Hidden units')
    plt.ylabel('Average Time (s)')
    plt.show()
