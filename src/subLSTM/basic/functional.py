import torch
import torch.nn.functional as F

def sublstm(input, hidden, input_layer, recurrent_layer):
    h_tm1, c_tm1 = hidden
    
    print("python sublstm old_h size: {}".format(h_tm1.size()))
    print("python sublstm old_cell size: {}".format(c_tm1.size()))
    
    proj_input = torch.sigmoid(input_layer(input) + recurrent_layer(h_tm1))

    in_gate, out_gate, z_t, f_gate = proj_input.chunk(4, 1)
    
    print("python sublstm in_gate size: {}".format(in_gate.size()))
    print("python sublstm out_gate size: {}".format(out_gate.size()))
    print("python sublstm z_t size: {}".format(z_t.size()))
    print("python sublstm f_gate size: {}".format(f_gate.size()))

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate
    
    print("python sublstm h_t (output) size: {}".format(h_t.size()))
    print("python sublstm c_t (cell) size: {}".format(c_t.size()))
    
    return h_t, c_t

def fsublstm(input, hidden, input_layer, recurrent_layer, f_gate):
    h_tm1, c_tm1 = hidden
    proj_input = torch.sigmoid(input_layer(input) + recurrent_layer(h_tm1))

    in_gate, out_gate, z_t = proj_input.chunk(3, 1)
    f_gate = f_gate.clamp(0, 1)

    c_t = c_tm1 * f_gate + z_t - in_gate
    h_t = torch.sigmoid(c_t) - out_gate

    return h_t, c_t
