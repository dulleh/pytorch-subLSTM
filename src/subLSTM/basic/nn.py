import os
import math
from itertools import product
from timeit import default_timer as timer

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.rnn import RNNCellBase
# from torch.nn.modules.rnn import PackedSequence
# from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as pad
from torch.autograd import Function
from .functional import sublstm, fsublstm
from collections import namedtuple
from typing import List, Tuple, Optional

path_to_sublstm_cuda_so = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'sublstm_cuda.so'))
torch.ops.load_library(path_to_sublstm_cuda_so)

### Example from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm.py
### See https://pytorch.org/docs/master/notes/extending.html for notes on autograd.Function
class SubLSTMFunction(Function):
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell, non_vars):
        X = non_vars[0]
        gate_weights = non_vars[1]

        new_h, new_cell, stacked_intermediates = torch.ops.sublstm.forward(input, gate_weights, old_h, old_cell)

        input_gate, output_gate, forget_gate, candidate_cell = stacked_intermediates.unbind()

        ctx.varies = [X, gate_weights, old_h, old_cell, forget_gate, candidate_cell, weights, new_cell, input_gate, output_gate]

        return new_h, new_cell

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        #grad_h = grad_h.contiguous()

        saved_data = ctx.varies
        X = saved_data[0]
        gate_weights = saved_data[1]
        old_h = saved_data[2]
        old_cell = saved_data[3]
        forget_gate = saved_data[4]
        candidate_cell = saved_data[5]
        weights = saved_data[6]
        new_cell = saved_data[7]
        input_gate = saved_data[8]
        output_gate = saved_data[9]

        outputs = torch.ops.sublstm.backward(grad_h, grad_cell, new_cell, input_gate, output_gate, forget_gate, candidate_cell, X, gate_weights, weights, old_cell)

        # Fix memory leak.
        del ctx.varies

        d_old_h = outputs[0]
        d_input = outputs[1]
        d_weights = outputs[2]
        d_bias = outputs[3]
        d_old_cell = outputs[4]
        d_gates = outputs[5]

        return d_input, d_weights, d_bias, d_old_h, d_old_cell, None

class SubLSTMCudaCell(nn.Module):
    def __init__(self, input_size, state_size, bias=True):
        super(SubLSTMCudaCell, self).__init__()
        self.input_size = input_size
        self.state_size = state_size

        gate_size = 4 * state_size
        input_layer = nn.Linear(input_size, gate_size, bias=bias)
        recurrent_layer = nn.Linear(state_size, gate_size, bias=False)

        # do this to have the same initialisation as non-cuda sublstm (for testing correctness)
        # ORDER is important!!
        """
        input_layer.reset_parameters()
        recurrent_layer.reset_parameters()
        """
        input_layer.reset_parameters()
        recurrent_layer.reset_parameters()

        input_weights = input_layer.weight.data.cuda()
        input_bias = input_layer.bias.data.cuda()
        recurrent_weights = recurrent_layer.weight.data.cuda()
        weightz = torch.cat((recurrent_weights, input_weights), 1)

        self.weights = nn.Parameter(weightz)
        self.bias = nn.Parameter(input_bias) if bias else None

        self.reset_parameters()
        self.flattenParameters()

    def flattenParameters(self):
        pass

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass


    def forward(self,input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        old_h, old_cell = state

        #Use .detach() because torch.no_grad() is not supported by TorchScript
        X = torch.cat((old_h.detach(), input.detach()), 1)
        X = X.detach()
        gate_weights = torch.addmm(self.bias.detach(), X, self.weights.detach().transpose(0, 1))

        # TorchScript version
        # still need to pass in all parameters so they can be saved for the backwards pass
        #new_h, new_cell = torch.ops.sublstm.apply(input, self.weights, self.bias, old_h, old_cell, [X, gate_weights])
        #return new_h, new_cell

        # If you don't want to use TorchScript, use this:
        #  It's also useful for debugging the C++ version of subLSTMFunction
        #  so long as you keep both up to date
        return SubLSTMFunction.apply(input, self.weights, self.bias, old_h, old_cell, [X, gate_weights])


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

    def flattenParameters(self):
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

def init_stacked_lstm(num_layers, layer, cell, input_size, hidden_size):
    layers = [layer(cell, input_size, hidden_size)] + \
             [layer(cell, hidden_size, hidden_size) for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)

# noinspection PyShadowingBuiltins,PyPep8Naming
class SubLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True,
                    cell_type='vanilla', batch_first=False, dropout=0.0, batch_size=1):
        super().__init__()

        self.streams = None
        self.num_streams = 0
        #self.times = []
        #self.epochtimes = []
        #self.backwardtimes = []
        #self.epochbackwardtimes = []

        self.totalforwardtime = 0
        self.totalbackwardtime = 0

        #self.memoryrecords = []
        #self.cachedmemoryrecords = []
        #self.epochmemory = []
        #self.epochcachedmemory = []

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

        layers = []
        for layer_num in range(num_layers):
            layer_in_size = input_size if layer_num == 0 else hidden_size
            layer_out_size = hidden_size

            layer = layer_type(layer_in_size, layer_out_size, bias)
            layers.append(layer)

            """
            #Tracing. But tracing doesn't work with a custom Function defined in Python.

            INPUT = torch.rand(batch_size, layer_in_size).cuda() if not batch_first else torch.rand(layer_in_size, batch_size).cuda()
            OLD_H = torch.rand(batch_size, hidden_size).cuda() if not batch_first else torch.rand(hidden_size, batch_size).cuda()
            OLD_CELL = torch.rand(batch_size, hidden_size).cuda() if not batch_first else torch.rand(hidden_size, batch_size).cuda()
            STATE = (OLD_H, OLD_CELL)
            traced_layer = torch.jit.trace(layer, (INPUT,STATE))
            layers.append(traced_layer)
            """
            """
            #Scripting.
            with torch.jit.optimized_execution(True):
                torch.backends.cudnn.benchmark = True
                traced_layer = torch.jit.script(layer_type(layer_in_size, layer_out_size, bias)).cuda()
            print(traced_layer.code)
            layers.append(traced_layer)
            """

        self.all_layers = nn.ModuleList(layers)

        if dropout > 0:
            self.use_dropout = True
            self.dropout = nn.Dropout(dropout)
        else:
            self.use_dropout = False

        self.flatten_parameters()
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    def flatten_parameters(self):
        #for module in self.children():
        #    module.flattenParameters()
        pass

    def forward(self,
            input: Tensor,
            states: Optional[List[Tuple[Tensor, Tensor]]]
            ) -> Tuple[Tensor, List[Tuple[Tensor, Tensor]]]:
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

        hx: List[Tuple[Tensor, Tensor]] = []
        if states is None:
            for l in range(self.num_layers):
                # use input.new_zeros so dtype and device are the same as the input's
                #hidden = input.new_zeros((max_batch_size, self.hidden_size), requires_grad=False)
                hidden = input.new_zeros((max_batch_size, self.hidden_size)).detach()
                hx.append((hidden, hidden))
        else:
            hx = states

        if self.batch_first:
            input = input.transpose(0, 1).contiguous()

        timesteps = input.size(0)

        """
        outputs = [input[i] for i in range(timesteps)]

        for time in range(timesteps):
            for l, layer in enumerate(self.all_layers):
                #layer = self.all_layers[l]

                #self.flatten_parameters()

                out, c = layer(outputs[time], hx[l])

                #self.memoryrecords.append(torch.cuda.memory_allocated() / 1024**2)
                #self.cachedmemoryrecords.append(torch.cuda.memory_cached() / 1024**2)

                if self.use_dropout:
                    out = self.dropout(out)

                hx[l] = (out, c)
                outputs[time] = out

        """

        # update this so that we add streams if we need to
        if self.streams is None:
            self.num_streams = min(timesteps, self.num_layers)
            self.streams = [torch.cuda.Stream() for i in range(self.num_streams)]

        # notice these are indexed oppositely so that the return values
        # are most efficiently and easily done
        outputGrid = [[None]*(timesteps) for i in range(self.num_layers)]
        stateGrid = [[None]*(self.num_layers) for i in range(timesteps)]

        diagStartL, diagStartT = 0, 0

        while True:
            torch.cuda.synchronize()
            currentL = diagStartL
            currentT = diagStartT

            streamIndex = 0

            while currentL < self.num_layers and currentT >= 0:
                currentStream = self.streams[streamIndex % self.num_streams]
                with torch.cuda.stream(currentStream):

                    if currentL == 0:
                        prevOutput = input[currentT]
                    else:
                        prevOutput = outputGrid[currentL-1][currentT]

                    if currentT == 0:
                        prevState = hx[currentL]
                    else:
                        prevState = stateGrid[currentT-1][currentL]

                    out, c = self.all_layers[currentL](prevOutput, prevState)

                    if self.use_dropout:
                        out = self.dropout(out)

                    stateGrid[currentT][currentL] = (out,c)
                    outputGrid[currentL][currentT] = out

                currentL += 1
                currentT -= 1
                streamIndex += 1

            if diagStartT != timesteps - 1: # moving across stage
                diagStartT += 1
            elif diagStartL != self.num_layers - 1: # moving up stage
                diagStartL += 1
            else: # on the last cell, so don't have any more to do
                break

        torch.cuda.synchronize()

        hx = stateGrid[timesteps - 1]
        outputs = outputGrid[self.num_layers - 1]

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
