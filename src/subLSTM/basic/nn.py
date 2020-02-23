import os
import math
from itertools import product

import torch
import torch.nn as nn
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.rnn import RNNCellBase
# from torch.nn.modules.rnn import PackedSequence
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad
from torch.autograd import Function
from .functional import sublstm, fsublstm
from torch.utils.cpp_extension import load

### Example from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm.py
### See https://pytorch.org/docs/master/notes/extending.html for notes on autograd.Function
class SubLSTMFunction(Function):
    path_to_this = os.path.abspath(os.path.dirname(__file__))
    cpp_path = os.path.join(path_to_this, "sublstm.cpp")
    cu_path = os.path.join(path_to_this, "sublstm.cu")

    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        forward_cpp = load(name="forward", sources=[SubLSTMFunction.cpp_path, SubLSTMFunction.cu_path])

        outputs = forward_cpp.forward(input,
                                      weights,
                                      bias,
                                      old_h,
                                      old_cell)
        new_h, new_cell = outputs[:2]

        ctx.varies = outputs[1:] + [weights] + [old_cell]

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        backward_cpp = load(name="backward", sources=[SubLSTMFunction.cpp_path, SubLSTMFunction.cu_path])

        grad_h.cuda()
        grad_cell.cuda()
        for i, sv in enumerate(ctx.varies):
            sv.cuda()

        outputs = backward_cpp.backward(
            grad_h.contiguous(),
            grad_cell.contiguous(),
            ctx.varies[0].contiguous(),
            ctx.varies[1].contiguous(),
            ctx.varies[2].contiguous(),
            ctx.varies[3].contiguous(),
            ctx.varies[4].contiguous(),
            ctx.varies[5].contiguous(),
            ctx.varies[6].contiguous(),
            ctx.varies[7].contiguous(),
            ctx.varies[8].contiguous()
            )

        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class SubLSTMCudaCell(nn.Module):
    def __init__(self, input_size, state_size, bias=True):
        super(SubLSTMCudaCell, self).__init__()
        self.input_size = input_size

        # do this to have the same initialisation as non-cuda sublstm (for testing)
        gate_size = 4 * state_size
        input_layer = nn.Linear(input_size, gate_size, bias=bias)
        recurrent_layer = nn.Linear(state_size, gate_size, bias=False)
        # ORDER is important!!
        input_layer.reset_parameters()
        recurrent_layer.reset_parameters()
        input_layer.reset_parameters()
        recurrent_layer.reset_parameters()

        input_weights = input_layer.weight.data
        input_bias = input_layer.bias.data
        recurrent_weights = recurrent_layer.weight.data
        weightz = torch.cat((recurrent_weights, input_weights), 1)

        self.state_size = state_size
        self.weights = nn.Parameter(weightz)
        self.bias = nn.Parameter(input_bias) if bias else None

        self.reset_parameters()
        #self.flatten_parameters()


    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def forward(self, input, state):
        return SubLSTMFunction.apply(input, self.weights, self.bias, *state)


class SubLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SubLSTMCell, self).__init__()

        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        gate_size = 4 * hidden_size

        self.input_layer = nn.Linear(input_size, gate_size, bias=bias)
        self.recurrent_layer = nn.Linear(hidden_size, gate_size, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def forward(self, input: torch.Tensor, hx):
        return sublstm(
            input, hx,
            self.input_layer,
            self.recurrent_layer,
        )


class fixSubLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(fixSubLSTMCell, self).__init__()
        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        gate_size = 3 * hidden_size

        self.input_layer = nn.Linear(input_size, gate_size, bias=bias)
        self.recurrent_layer = nn.Linear(hidden_size, gate_size, bias=bias)
        self.f_gate = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def forward(self, input, hx):
        return fsublstm(
            input, hx,
            self.input_layer,
            self.recurrent_layer,
            self.f_gate
        )


# noinspection PyShadowingBuiltins,PyPep8Naming
class SubLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                    cell_type='vanilla', batch_first=False, dropout=0.0):
        super().__init__()

        # Uncomment to get layers of different size. Disable for consistency with LSTM
        # if isinstance(hidden_size, list) and len(hidden_size) != num_layers:
        #     raise ValueError(
        #         'Length of hidden_size list is not the same as num_layers.'
        #         'Expected {0} got {1}'.format(
        #             num_layers, len(hidden_size))
        #     )

        # if isinstance(hidden_size, int):
        #     hidden_size = [hidden_size] * num_layers

        # Some python "magic" to assign all parameters as class attributes
        self.__dict__.update(locals())

        # Use for bidirectional later
        self.cell_type = cell_type
        suffix = ''
        if cell_type == 'fixed_forget':
            #layer_type = fixSubLSTMCell
            layer_type = SubLSTMCell
        elif cell_type == 'cuda':
            layer_type = SubLSTMCudaCell
        elif cell_type == 'vanilla':
            layer_type = SubLSTMCell
        else:
            raise Exception('cell_type must one of \'vanilla\', \'fixed_forget\', or \'cuda\'. Recieved: {}'.format(cell_type))

        for layer_num in range(num_layers):

            layer_in_size = input_size if layer_num == 0 else hidden_size
            layer_out_size = hidden_size

            layer = layer_type(layer_in_size, layer_out_size, bias)

            self.add_module('layer_{}'.format(layer_num + 1), layer)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

        else:
            self.dropout = False

        self.flatten_parameters()
        self.reset_parameters()

    @property
    def all_weights(self):
        return [[getattr(self, name) for name in param_names]
            for param_names in self._all_params]

    @property
    def all_layers(self):
        return [getattr(self, 'layer_{}'.format(layer + 1)) for layer in range(self.num_layers)]

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def flatten_parameters(self):
        pass

    #@staticmethod
    def forward(self, input, hx=None):
        # TODO: Check docs later and add the packed sequence and seq2seq models
        # is_packed = isinstance(input, PackedSequence)
        #
        # if is_packed:
        #     input, batch_size = pad(input)
        #     max_batch_size = batch_size[0]
        # else:
        #     batch_size = None
        #     max_batch_size = input.size(0) if self.batch_first else input.size(1)

        max_batch_size = input.size(0) if self.batch_first else input.size(1)

        if hx is None:
            hx = []
            for l in range(self.num_layers):
                # use input.new_zeros so dtype and device are the same as the input's
                hidden = input.new_zeros(
                    (max_batch_size, self.hidden_size), requires_grad=False)
                hx.append((hidden, hidden))

        if self.batch_first:
            input = input.transpose(0, 1).contiguous()
        print(input.device)

        timesteps = input.size(0)
        outputs = [input[i] for i in range(timesteps)]
        all_layers = self.all_layers

        for time, l in product(range(timesteps), range(self.num_layers)):
            layer = all_layers[l]

            out, c = layer(outputs[time], hx[l])

            if self.dropout:
                out = self.dropout(out)

            hx[l] = (out, c)
            outputs[time] = out

        out = torch.stack(outputs)
        if self.batch_first:
            out = out.transpose(0, 1)

        # TODO: Check docs later and add the packed sequence option
        # if is_packed:
        #     out = pack(out, batch_size)

        return out, hx

    def _apply(self, fn):
        ret = super(SubLSTM, self)._apply(fn)
        self.flatten_parameters()
        return ret

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if self.num_layers != 1:
            s += ', num_layers={num_layers}'
        if self.bias is not True:
            s += ', bias={bias}'
        if self.batch_first is not False:
            s += ', batch_first={batch_first}'
        # if self.dropout != 0:
        #     s += ', dropout={dropout}'
        # if self.bidirectional is not False:
        #     s += ', bidirectional={bidirectional}'
        return s.format(**self.__dict__)
