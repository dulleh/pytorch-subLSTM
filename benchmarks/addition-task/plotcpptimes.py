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
from utils import train, test

import matplotlib.pyplot as plt

def drawepochs(epochs, epochsb, model_name):
    sizeofepoch = int(len(epochs)/5)
    fig, axs = plt.subplots(5, 2, constrained_layout=True)
    fig.suptitle('Times Across Epochs for {}'.format(model_name))
    for i in range(0,5):
        axs[i][0].bar(list(range(0,sizeofepoch)), epochs[i*sizeofepoch : (i+1)*sizeofepoch])
        axs[i][0].set(xlabel='total forward() time for epoch {} was {}s'.format(i, sum(epochs[i*sizeofepoch :(i+1)*sizeofepoch])), ylabel='Time (s)')
        axs[i][1].bar(list(range(0,sizeofepoch)), epochsb[i*sizeofepoch : (i+1)*sizeofepoch])
        axs[i][1].set(xlabel='total backward() time for epoch {} was {}s'.format(i, sum(epochsb[i*sizeofepoch : (i+1)*sizeofepoch])), ylabel='Time (s)')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Addition task')
    path_to_this = os.path.abspath(os.path.dirname(__file__))
    cuda_file_name = 'functionforwardtime.csv'
    cuda_save_path = os.path.join(path_to_this, cuda_file_name)

    cudatimes = np.loadtxt(cuda_save_path, delimiter=',')
    drawepochs(cudatimes, cudatimes, "sublstm cuda")
