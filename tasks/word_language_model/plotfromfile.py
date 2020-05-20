# coding: utf-8

import sys
import os
import argparse
import time
import csv
import math

import numpy as np
import pickle

sys.path.insert(0, '../../src/')
sys.path.insert(0, '../')

import matplotlib.pyplot as plt

if __name__ == '__main__':
    path_to_this = os.path.abspath(os.path.dirname(__file__))

    lstm_file_path = os.path.join(path_to_this, 'PTB_TIMES_LSTM.csv')
    sublstm_file_path = os.path.join(path_to_this, 'PTB_TIMES_subLSTMCuda.csv')
    lstm_file = open(lstm_file_path, 'rb')
    sublstm_file = open(sublstm_file_path, 'rb')
    lstm_data = pickle.load(lstm_file)
    sublstm_data = pickle.load(sublstm_file)

    lstm_file_path2 = os.path.join(path_to_this, 'PTB_TIMES2_LSTM.csv')
    sublstm_file_path2 = os.path.join(path_to_this, 'PTB_TIMES2_subLSTMCuda.csv')
    lstm_file2 = open(lstm_file_path2, 'rb')
    sublstm_file2 = open(sublstm_file_path2, 'rb')
    lstm_data2 = pickle.load(lstm_file2)
    sublstm_data2 = pickle.load(sublstm_file2)


    lstm_file_path3 = os.path.join(path_to_this, 'PTB_TIMES3_LSTM.csv')
    sublstm_file_path3 = os.path.join(path_to_this, 'PTB_TIMES3_subLSTMCuda.csv')
    lstm_file3 = open(lstm_file_path3, 'rb')
    sublstm_file3 = open(sublstm_file_path3, 'rb')
    lstm_data3 = pickle.load(lstm_file3)
    sublstm_data3 = pickle.load(sublstm_file3)

    python_file_path = os.path.join(path_to_this, 'PTB_TIMES_1EPOCH_subLSTM.csv')
    python_file = open(python_file_path, 'rb')
    python_data = pickle.load(python_file)

    #batch_sizes = [4, 16]
    batch_sizes = [4, 16, 128]
    #input_sizes = [50, 128]
    input_sizes = [64, 512]
    #hidden_sizes = range(50, 100 + 1, 50)
    hidden_sizes = range(50, 1050 + 1, 100)

    handles, labels = None, None

    fig, axs = plt.subplots(len(batch_sizes), len(input_sizes), constrained_layout=True, figsize=(10,12))
    fig.suptitle('Avg. Per-Epoch Times Across Parameter Sizes')

    for b, batch_size in enumerate(batch_sizes):
        for i, input_size in enumerate(input_sizes):
            lstm_vals = lstm_data[b][i]
            lstm_avgs = [t for (t,v) in lstm_vals]
            lstm_stds = [v for (t,v) in lstm_vals]
            axs[b][i].plot(list(hidden_sizes), lstm_avgs, label="cuDNN LSTM (run 1)", color='orange', linestyle='--', alpha=0.5)

            lstm_vals2 = lstm_data2[b][i]
            lstm_avgs2 = [t for (t,v) in lstm_vals2]
            lstm_stds2 = [v for (t,v) in lstm_vals2]
            axs[b][i].plot(list(hidden_sizes), lstm_avgs2, label="cuDNN LSTM (run 2)", color='orange', linestyle=':', alpha=0.5)

            lstm_vals3 = lstm_data3[b][i]
            lstm_avgs3 = [t for (t,v) in lstm_vals3]
            lstm_stds3 = [v for (t,v) in lstm_vals3]
            axs[b][i].plot(list(hidden_sizes), lstm_avgs3, label="cuDNN LSTM (run 3)", color='orange', linestyle='-.', alpha=0.5)

            lstm_avgs_final = (np.array(lstm_avgs) + np.array(lstm_avgs2) + np.array(lstm_avgs3)) / 3
            axs[b][i].plot(list(hidden_sizes), lstm_avgs_final, label="cuDNN LSTM Avg.", color='orange', linestyle='-')

            sublstm_vals = sublstm_data[b][i]
            sublstm_avgs = [t for (t,v) in sublstm_vals]
            sublstm_stds = [v for (t,v) in sublstm_vals]
            axs[b][i].plot(list(hidden_sizes), sublstm_avgs, label="CUDA subLSTM (run 1)",  color='blue', linestyle='--', alpha=0.5)

            sublstm_vals2 = sublstm_data2[b][i]
            sublstm_avgs2 = [t for (t,v) in sublstm_vals2]
            sublstm_stds2 = [v for (t,v) in sublstm_vals2]
            axs[b][i].plot(list(hidden_sizes), sublstm_avgs2, label="CUDA subLSTM (run 2)", color='blue', linestyle='-.', alpha=0.5)

            sublstm_vals3 = sublstm_data3[b][i]
            sublstm_avgs3 = [t for (t,v) in sublstm_vals3]
            sublstm_stds3 = [v for (t,v) in sublstm_vals3]
            axs[b][i].plot(list(hidden_sizes), sublstm_avgs3, label="CUDA subLSTM (run 3)", color='blue', linestyle=':')

            sublstm_avgs_final = (np.array(sublstm_avgs) + np.array(sublstm_avgs2) + np.array(sublstm_avgs3)) / 3
            axs[b][i].plot(list(hidden_sizes), sublstm_avgs_final, label="CUDA subLSTM Avg.", color='blue', linestyle='-')

            python_vals = python_data[b][i]
            python_avgs = [t for (t,v) in python_vals]
            axs[b][i].plot(list(hidden_sizes), python_avgs, label="Python subLSTM (run 1)", color='green', linestyle='--', alpha=0.5)

            axs[b][i].set_ylim(ymin=0)
            axs[b][i].set(xlabel='Hidden units', ylabel='Time (s)')
            axs[b][i].title.set_text('Batch Size {}, Input Size {}'.format(batch_size, input_size))

            handles, labels = axs[b][i].get_legend_handles_labels()


    fig.legend(handles, labels, loc='upper left')

    plt.savefig(os.path.join(path_to_this, 'finalbenchmarks.png'))


    lstm_file.close()
    sublstm_file.close()
