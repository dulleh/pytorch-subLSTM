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

def drawtimevshidden(pythontimes, cudatimes, numepochs, batchsize, seqlen, trainingsize):
    plt.suptitle('Avg. Forward Time per Epoch Vs Hidden Units with Batch Size {}, Seq. Length {}, Training Size {} across {} epochs'.format(batchsize, seqlen, trainingsize, numepochs))
    plt.plot(np.arange(1, len(pythontimes)+1, 1), pythontimes, label='Python')
    plt.plot(np.arange(1, len(pythontimes)+1, 1), cudatimes, label='CUDA forward, C++ backward')
    plt.xlabel('Hidden units')
    plt.ylabel('Average Time (s)')
    plt.show()

class BatchGenerator:
    def __init__(self, training_size, batch_size, min_arg, max_arg, seq_len, num_addends):
        self.min_arg = min_arg
        self.max_arg = max_arg
        self.batch_size = batch_size
        self.training_size = training_size
        self.seq_len = seq_len
        self.num_addends = num_addends

    def __iter__(self):
        for _ in range(len(self)):
            yield self.next_batch()

    def __len__(self):
        return self.training_size // self.batch_size

    def next_batch(self):
        batch_size, min_arg, max_arg, seq_len = self.batch_size, self.min_arg, self.max_arg, self.seq_len

        random_state = np.random.RandomState(seed=12)

        inputs = random_state.uniform(
            low=min_arg, high=max_arg, size=(batch_size, seq_len, 2))
        inputs[:, :, 1] = 0

        # Neat trick to sample the positions to unmask
        mask = random_state.rand(batch_size, seq_len).argsort(axis=1)[:,:self.num_addends]
        mask.sort(axis=1)

        # Mask is in the wrong shape (batch_size, num_addends) for slicing
        inputs[range(batch_size), mask.T, 1] = 1
        targets = np.sum(inputs[:, :, 0] * (inputs[:, :, 1]), axis=1).reshape(-1, 1)

        inputs = torch.as_tensor(inputs, dtype=torch.float, device='cuda')
        targets = torch.as_tensor(targets, dtype=torch.float, device='cuda')

        return inputs, targets

########################################################################################
# PARSE THE INPUT
########################################################################################

def main(args):
    ########################################################################################
    # SETTING UP THE DEVICE AND SEED
    ########################################################################################
    input_size, hidden_size, responses = 2, args.nhid, 1

    pythonforwardtimes = []
    cudaforwardtimes = []

    for thismodel in ['subLSTM', 'subLSTMCuda']:
        # will be
        # [total time across epochs>=1 for hid=1, total time hid=2, total time hid=3, ..., total time hid=nhid]
        forwardtimes = []

        # We reinterpret 'nhid' parameter to be the maximum nhid to loop up to
        for current_hidden_size in range(1,hidden_size):
            torch.cuda.empty_cache()

            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            # additional parameters that need to be removed if you want non-deterministic behaviour
            # https://pytorch.org/docs/stable/notes/randomness.html
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            if args.cuda and torch.cuda.is_available():
                torch.cuda.manual_seed(args.seed)
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')

            ########################################################################################
            # CREATE THE DATA GENERATOR
            ########################################################################################

            seq_len, num_addends = args.seq_length, args.num_addends
            min_arg, max_arg = args.min_arg, args.max_arg
            N, batch_size, test_size = args.training_size, args.batch_size, args.testing_size

            train_size = int(N * (1 - args.train_val_split))
            val_size = N - train_size

            training_data = BatchGenerator(train_size, batch_size, min_arg, max_arg, seq_len, num_addends)
            validation_data = BatchGenerator(val_size, val_size, min_arg, max_arg, seq_len, num_addends)
            test_data = BatchGenerator(test_size, test_size, min_arg, max_arg, seq_len, num_addends)

            ########################################################################################
            # CREATE THE MODEL
            ########################################################################################

            model = init_model(
                model_type=thismodel,
                n_layers=args.nlayers, hidden_size=current_hidden_size,
                input_size=input_size, output_size=responses, class_task=False,
                device=device,
                dropout=args.dropout,
                script=args.script
            )

            ########################################################################################
            # SET UP OPTIMIZER & OBJECTIVE FUNCTION
            ########################################################################################

            optimizer = optim.RMSprop(model.parameters(),
                lr=args.lr, eps=1e-10, weight_decay=args.l2_norm, momentum=0.9)

            criterion = nn.MSELoss()

            ########################################################################################
            # TRAIN MODEL
            ########################################################################################

            epochs, log_interval = args.epochs, args.log_interval
            loss_trace, best_loss = [], np.inf

            forwardtimesum = []



            if args.verbose:
                print('Training {} model with parameters:'
                        '\n\tnumber of layers: {}'
                        '\n\thidden units: {}'
                        '\n\tmax epochs: {}'
                        '\n\tbatch size: {}'
                        '\n\toptimizer: {}, lr={}, l2={}'.format(
                            args.model, args.nlayers, current_hidden_size, epochs,
                            batch_size, args.optim, args.lr, args.l2_norm
                        ))

            try:
                for e in range(epochs):

                    # Train model for 1 epoch over whole dataset
                    epoch_trace = train(
                        model=model, data_loader=training_data,
                        criterion=criterion, optimizer=optimizer, grad_clip=args.clip,
                        log_interval=log_interval,
                        device=device,
                        track_hidden=args.track_hidden,
                        verbose=args.verbose
                    )

                    # skip the first epoch as that is a special case
                    if e > 0:
                        forwardtimesum.append(model.rnn.totalforwardtime)
                    model.rnn.totalforwardtime = 0

            except KeyboardInterrupt:
                if args.verbose:
                    print('Keyboard interruption. Terminating training.')

            # average the times over epochs (skipping the first)
            forwardtimes.append(np.mean(forwardtimesum))

        if thismodel == 'subLSTM':
            pythonforwardtimes.extend(forwardtimes)
        elif thismodel == 'subLSTMCuda':
            cudaforwardtimes.extend(forwardtimes)

    drawtimevshidden(pythonforwardtimes, cudaforwardtimes, args.epochs - 1, args.batch_size, args.seq_length, args.training_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Addition task')

    # Model parameters
    parser.add_argument('--model', type=str, default='subLSTM',
        help='RNN model to use. One of subLSTM|fix-subLSTM|LSTM|GRU|subLSTMCuda')
    parser.add_argument('--nlayers', type=int, default=1,
        help='number of layers')
    parser.add_argument('--nhid', type=int, default=50,
        help='number of hidden units per layer')
    parser.add_argument('--dropout', type=float, default=0.0,
        help='drop probability for Bernoulli Dropout')
    parser.add_argument('--gact', type=str, default='relu',
        help='gate activation function relu|sig')
    parser.add_argument('--gbias', type=float, default=0,
        help='gating bias')
    parser.add_argument('--script', action='store_true', help='Use TorchScript version')

    # Data parameters
    parser.add_argument('--seq-length', type=int, default=50,
        help='sequence length')
    parser.add_argument('--num-addends', type=int, default=2,
        help='the number of addends to be unmasked in each sequence'
        'must be less than the sequence length')
    parser.add_argument('--min-arg', type=float, default=0,
        help='minimum value of the addends')
    parser.add_argument('--max-arg', type=float, default=1,
        help='maximum value of the addends')
    parser.add_argument('--training-size', type=int, default=10000,
        help='size of the randomly created training set')
    parser.add_argument('--testing-size', type=int, default=10000,
        help='size of the randomly created test set')
    parser.add_argument('--train-val-split', type=float, default=0.2,
        help='proportion of trainig data used for validation')
    parser.add_argument('--batch-size', type=int, default=50, metavar='N',
        help='batch size')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=40,
        help='max number of training epochs')
    parser.add_argument('--optim', type=str, default='rmsprop',
        help='gradient descent method,'
        'supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
    parser.add_argument('--lr', type=float, default=1e-4,
        help='initial learning rate')
    parser.add_argument('--l2-norm', type=float, default=0,
        help='weight of L2 norm')
    parser.add_argument('--clip', type=float, default=1,
        help='gradient clipping')
    parser.add_argument('--track-hidden', action='store_true',
        help='keep the hidden state values across a whole epoch of training')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
        help='report interval')

    # Replicability and storage
    parser.add_argument('--save', type=str,  default='results',
        help='path to save the final model')
    parser.add_argument('--seed', type=int, default=18092,
        help='random seed')

    # CUDA
    parser.add_argument('--cuda', action='store_true',
        help='use CUDA')

    # Print options
    parser.add_argument('--verbose', action='store_true',
        help='print the progress of training to std output.')
    parser.add_argument('--timing', action='store_true',
        help='print average training times')

    # Testing
    parser.add_argument('--test', action='store_true',
        help='test the loss trace against the relevant cached example')

    args = parser.parse_args()
    main(args)
