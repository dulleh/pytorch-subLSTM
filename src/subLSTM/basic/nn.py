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
    # Remember that old_h is the output from the previous cell i.e. h_(t-1)
    # Likewise, old_cell is C_(t-1). Both of these get updated every time step
    # The parameter weights, however, stays the same until it is updated in the
    # backwards pass. It contains 4 gates: input_gate, output_gate,
    #                                      input_tanh_gate, forget_gate
    @staticmethod
    def forward(ctx, input, weights, bias, old_h, old_cell):
        ## Need to see what @staticmethod keyword does..
        ## and where to move the path stuff if wanting to only do it once
        # Load/Compile the c++/cuda files
        #"""
        print("old_h size: ", old_h.size()) # [20,50]
        print("input size:", input.size()) # [20, 2]
        X = torch.cat((old_h, input), 1)
        print("weights: ", weights.size()) # [200, 52]
        print("X: ", X.size()) # [20, 52]
        print("bias: ", bias.size()) # [200]
        gate_weights = bias + X.mm(weights.t()) # [20, 200]
        print("gate_weights", gate_weights.size())
        batch_size = old_cell.size(0) # 20
        state_size = old_cell.size(1) # 50
        gates = torch.sigmoid(gate_weights.reshape(batch_size, 4, state_size)) # [20, 5, 50]
        
        in_gate, out_gate, z_t, f_gate = gates.chunk(4, 1)
        in_gate = in_gate.squeeze()
        print("in_gate", in_gate.size())
        print("out_gate", out_gate.size())
        print("z_t", z_t.size())
        print("f_gate", f_gate.size())
        c_t = old_cell * f_gate + z_t - in_gate
        h_t = torch.sigmoid(c_t) - out_gate
        
        variables = [c_t] + [in_gate] + [out_gate] + [f_gate] + [z_t] + [X] + [gates] + [weights] + [old_cell]
        ctx.save_for_backward(*variables)
        
        print("output h_t: ", h_t.size())
        print("c_t: ", c_t.size())
        print("old_cell: ", old_cell.size())
		
        return h_t, c_t
        """
        path_to_this = os.path.abspath(os.path.dirname(__file__))
        sublstm_cpp_path = os.path.join(path_to_this, "sublstm.cpp")
        sublstm_cu_path = os.path.join(path_to_this, "sublstm.cu")
        forward_cpp = load(name="forward", sources=[sublstm_cpp_path, sublstm_cu_path])
        # Perform forward pass
        ## TODO: look into .contiguous and how to use it less
        print("input size 0: ", input.size())
        input.contiguous()
        print("next: ", input.size())
        weights.contiguous()
        bias.contiguous()
        old_h.contiguous()
        old_cell.contiguous()
        ## Without this second call to .contiguous on input we get an error?
        outputs = forward_cpp.forward(input.contiguous(),
                                      weights,
                                      bias,
                                      old_h,
                                      old_cell)
        new_h, new_cell = outputs[:2]
        variables = outputs[1:] + [weights] + [old_cell]
        ctx.save_for_backward(*variables)

        return new_h, new_cell
        """

    @staticmethod
    def backward(ctx, grad_h, grad_cell):
        ## Need to see what @staticmethod keyword does..
        ## and where to move the path stuff if wanting to only do it once
        # Load/Compile the c++/cuda files
        path_to_this = os.path.abspath(os.path.dirname(__file__))
        sublstm_cpp_path = os.path.join(path_to_this, "sublstm.cpp")
        sublstm_cu_path = os.path.join(path_to_this, "sublstm.cu")
        print(sublstm_cpp_path)
        backward_cpp = load(name="backward", sources=[sublstm_cpp_path, sublstm_cu_path])
        print(grad_h.size())
        print(grad_cell.size())
        grad_h.cuda()
        grad_cell.cuda()
        print(grad_h.device)
        print(grad_cell.device)
        for i, sv in enumerate(ctx.saved_variables):
            print('saved_var[{}]_size {}'.format(i, sv.size()))
            sv.cuda()
        for i, sv in enumerate(ctx.saved_variables):
            print(sv.device)

        outputs = backward_cpp.backward(
            grad_h.contiguous(),
            grad_cell.contiguous(),
            *ctx.saved_variables
        )
        d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates = outputs
        return d_input, d_weights, d_bias, d_old_h, d_old_cell


class SubLSTMCudaCell(nn.Module):
    def __init__(self, input_size, state_size, bias=True):
        super(SubLSTMCudaCell, self).__init__()
        self.input_size = input_size
        #print("param: input_size: {}".format(input_size))
        #print("param: state_size: {}".format(state_size))
        self.state_size = state_size
        self.weights = nn.Parameter(
            torch.Tensor(4 * state_size, input_size + state_size))
        self.bias = nn.Parameter(torch.Tensor(4 * state_size)) if bias else None
        
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass
    
    #@staticmethod
    def forward(self, input, state):
        print('CELL input_size: {}'.format(input.size()))
        #print('CELL weights_size: {}'.format(self.weights.size()))
        #if self.bias is not None:
            #print('bias_size: {}'.format(self.bias.size()))
        for i, st in enumerate(state):
            print('state[{}]_size {}'.format(i, st.size()))
        return SubLSTMFunction.apply(input, self.weights, self.bias, *state)


class SubLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(SubLSTMCell, self).__init__()
        # Set the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        gate_size = 4 * hidden_size

        print("input_size: {}".format(input_size))
        print("state_size: {}".format(hidden_size))
        print("gate_size: {}".format(gate_size))

        self.input_layer = nn.Linear(input_size, gate_size, bias=bias)
        self.recurrent_layer = nn.Linear(hidden_size, gate_size, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.children():
            try:
                module.reset_parameters()
            except AttributeError:
                pass

    #@staticmethod
    def forward(self, input: torch.Tensor, hx):
        print('CELL input_size: {}'.format(input.size()))
        for i, st in enumerate(hx):
            print('state[{}]_size {}'.format(i, st.size()))
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

    #@staticmethod
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
        print(SubLSTM)
        print(self.__class__)
        print(isinstance(self, SubLSTM))
        
        #self.as_super = super(SubLSTM, self)
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
            input = input.transpose(0, 1)

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
