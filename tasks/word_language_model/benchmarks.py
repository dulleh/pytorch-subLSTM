# coding: utf-8
import sys
import os
import argparse
import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pickle

import data
import model

sys.path.insert(0, '../../src/')
sys.path.insert(0, '../')

path_to_this = os.path.abspath(os.path.dirname(__file__))

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default=os.path.join(path_to_this, 'data','penn'),
          help='location of the data corpus')
parser.add_argument('--model', type=str, default='subLSTMCuda',
          help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, subLSTM, subLSTMCuda)')
parser.add_argument('--emsize', type=int, default=200,
          help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
          help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
          help='number of layers')
parser.add_argument('--lr', type=float, default=0.0001,
          help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
          help='gradient clipping')
parser.add_argument('--optim', type=str, default='rmsprop',
          help='learning rule, supports adam|sparseadam|adamax|rmsprop|sgd|adagrad|adadelta')
parser.add_argument('--epochs', type=int, default=40,
          help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
          help='batch size')
parser.add_argument('--bptt', type=int, default=35,
          help='sequence length')
parser.add_argument('--dropout', type=float, default=0.5,
          help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
          help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
          help='random seed')
parser.add_argument('--cuda', action='store_true',
          help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
          help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
          help='path to save the final model')
args = parser.parse_args()
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
  if not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  else:
    torch.cuda.manual_seed(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz):
  # Work out how cleanly we can divide the dataset into bsz parts.
  nbatch = data.size(0) // bsz
  # Trim off any extra elements that wouldn't cleanly fit (remainders).
  data = data.narrow(0, 0, nbatch * bsz)
  # Evenly divide the data across the bsz batches.
  data = data.view(bsz, -1).t().contiguous()
  return data

eval_batch_size = 10

###############################################################################
# Build the model
###############################################################################
ntokens = len(corpus.dictionary)


criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def detach_hidden_state(hidden_state):
  if isinstance(hidden_state, torch.Tensor):
    return hidden_state.detach()
  elif isinstance(hidden_state, list):
    return [detach_hidden_state(h) for h in hidden_state]
  elif isinstance(hidden_state, tuple):
    return tuple(detach_hidden_state(h) for h in hidden_state)
  return None


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(source, i):
  seq_len = min(args.bptt, source.size(0) - 1 - i)
  data = source[i:i+seq_len]
  target = source[i+1:i+1+seq_len].view(-1)

  #print("input shape: ", data.shape)

  return data, target


def evaluate(data_source):
  # Turn on evaluation mode which disables dropout.
  model.eval()
  with torch.no_grad():
      total_loss = torch.zeros(1).cuda()
      ntokens = len(corpus.dictionary)
      #hidden = model.init_hidden(eval_batch_size)
      hidden = None
      for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i)

        output, hidden = model(data, hidden)

        total_loss += data.size(0) * criterion(output.reshape(-1, ntokens), targets).data
        hidden = detach_hidden_state(hidden)
      return total_loss.data[0] / len(data_source)

def train(train_data, ntokens, model):
  # Turn on training mode which enables dropout.
  model.train(True)
  total_loss = torch.zeros(1).cuda()
  hidden = None
  for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    data, targets = get_batch(train_data, i)

    optimizer.zero_grad()
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    hidden = detach_hidden_state(hidden)

    # forward
    output, hidden = model(data, hidden)

    loss = criterion(output.reshape(-1, ntokens), targets)

    loss.backward()

    # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    total_loss += loss.data

# Loop over epochs.
lr = args.lr
best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
try:
  path_to_this = os.path.abspath(os.path.dirname(__file__))

  all_times = []

  if args.cuda:
      corpus.train = corpus.train.cuda()

  #for batch_size in [4, 16]:
  for batch_size in [4, 16, 128]:
      this_batch_times = []

      train_data = batchify(corpus.train, batch_size)

      #for input_size in [50, 128]:
      for input_size in [64, 512]:

          this_input_times = []

          #for hidden_size in range(50, 100 + 1, 50):
          for hidden_size in range(50, 1050 + 1, 100):
              thismodel = model.RNNModel(args.model, ntokens, input_size, hidden_size, args.nlayers, 0, args.tied, batch_size)
              thismodel.cuda()
              optimizer = optim.SGD(thismodel.parameters(), lr=0.01)

              this_hidden_times = []

              for epoch in range(1, 5 + 1):
                torch.cuda.synchronize()
                epoch_start_time = time.time()

                train(train_data, ntokens, thismodel)

                torch.cuda.synchronize()
                elapsedTime = time.time() - epoch_start_time

                if epoch != 1:
                    this_hidden_times.append(elapsedTime)

              # only store the mean time and standard deviation
              this_input_times.append([np.mean(this_hidden_times), np.std(this_hidden_times)])

          this_batch_times.append(this_input_times)

      all_times.append(this_batch_times)
  save_path = os.path.join(path_to_this,  'PTB_TIMES3_{}.csv'.format(args.model))
  #np.savetxt(save_path, all_times, delimiter=',')
  output = open(save_path, 'wb')
  pickle.dump(all_times, output)
  output.close()

except KeyboardInterrupt:
  print('-' * 89)
  print('Exiting from training early')

# """
# # Load the best saved model.
# if args.model == 'subLSTMCuda':
#     #torch.jit.load(args.save)
#     with open(args.save, 'rb') as f:
#       model = torch.load(f)
# else:
#     with open(args.save, 'rb') as f:
#       model = torch.load(f)
#
#
# # Run on test data.
# if args.cuda:
#     corpus.test = corpus.test.cuda()
# test_data = batchify(corpus.test, eval_batch_size)
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#   test_loss, math.exp(test_loss)))
# print('=' * 89)
# """
