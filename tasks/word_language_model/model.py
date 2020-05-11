import sys
import os
import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Variable
from collections import namedtuple
from typing import List, Tuple, Optional

sys.path.insert(0, '../../src/')
sys.path.insert(0, '../')

from subLSTM.basic.nn import SubLSTM


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, batch_size=1):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            #self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=False)
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout, batch_first=False)
        elif rnn_type == 'subLSTM':
            self.rnn = SubLSTM(input_size=ninp,
                                 hidden_size=nhid,
                                 num_layers=nlayers,
                                 cell_type='vanilla',
                                 batch_first=False,
                                 dropout=dropout)
        elif rnn_type == 'subLSTMCuda':
            self.rnn = SubLSTM(input_size=ninp,
                                  hidden_size=nhid,
                                  num_layers=nlayers,
                                  cell_type='cuda',
                                  batch_first=False,
                                  dropout=dropout)
            """
            with torch.jit.optimized_execution(True):
                torch.backends.cudnn.benchmark = True
                self.rnn = torch.jit.script(SubLSTM(input_size=ninp,
                                      hidden_size=nhid,
                                      num_layers=nlayers,
                                      cell_type='cuda',
                                      batch_first=False,
                                      dropout=dropout))
            print(self.rnn.code)
            """
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,
            input: Tensor,
            hidden: Optional[List[Tuple[Tensor, Tensor]]]
            ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
        emb = self.drop(self.encoder(input))
        output, new_hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape(output.size(0)*output.size(1), output.size(2)))
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1))
        return decoded, new_hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        elif self.rnn_type in ['subLSTM','subLSTMCuda']:
            return None
        else:
            return Variable(weight.new(self.nlayers, bsz, self.nhid).zero_())
