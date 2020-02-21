/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * and taking guidance from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda.cpp
  */
//Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
#include <torch/extension.h>
#include <iostream>
#include <cassert>
//namespace py = pybind11;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

// tanh'(z) = 1 - tanh^2(z)
torch::Tensor d_tanh(torch::Tensor z) {
  return 1 - z.tanh().pow(2);
}

std::vector<torch::Tensor> forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> forward(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);

  return forward_cuda(input, weights, bias, old_h, old_cell);
}

std::vector<torch::Tensor> sublstm_backward(
    torch::Tensor grad_h,
    torch::Tensor grad_cell,
    torch::Tensor new_cell,
    torch::Tensor input_gate, // these are the outputs of these gates
    torch::Tensor output_gate,
    torch::Tensor forget_gate,
    torch::Tensor candidate_cell,
    torch::Tensor X,
    torch::Tensor gate_weights, // gate outputs, pre-activation
    torch::Tensor weights, // actual weights in the gates
    torch::Tensor old_cell) {
  CHECK_INPUT(grad_h);
  CHECK_INPUT(grad_cell);
  CHECK_INPUT(new_cell);
  CHECK_INPUT(input_gate);
  CHECK_INPUT(output_gate);
  CHECK_INPUT(forget_gate);
  CHECK_INPUT(candidate_cell);
  CHECK_INPUT(X);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(weights);
  CHECK_INPUT(old_cell);
  std::cout << "dE/dh" << grad_h << std::endl;
  std::cout << "grad_cell" << grad_cell << std::endl;
  std::cout << "new_cell" << new_cell << std::endl;
  std::cout << "input_gate" << input_gate << std::endl;
  std::cout << "output_gate" << output_gate << std::endl;
  std::cout << "candidate_cell" << candidate_cell << std::endl;
  std::cout << "X" << X << std::endl;
  std::cout << "gate_weights" << gate_weights << std::endl;
  std::cout << "weights" << weights << std::endl;
  std::cout << "old_cell" << old_cell << std::endl;

  torch::Tensor d_output_gate = -grad_h; // ht = sigmoid(ct) - ot (where ot is post activation)
  torch::Tensor d_new_cell = (grad_h * d_sigmoid(new_cell)) + (grad_cell * forget_gate); // needs to be f_(t+1)?
  // subLSTM definition that ht = sigmoid(ct) - ot so delta ct = delta ht * (dht/dct)

  torch::Tensor d_old_cell = d_new_cell * forget_gate; // dE/dct-1 = dE/dct * dct/dct-1 = delta(ct) * ft
  // is forget_gate = ft? - yes.
  torch::Tensor d_candidate_cell = d_new_cell; // this is delta(zt)
  torch::Tensor d_input_gate = -d_new_cell; // this is delta(it)
  torch::Tensor d_forget_gate = d_new_cell * old_cell;

  torch::Tensor d_gates =
        torch::cat({d_input_gate, d_output_gate, d_candidate_cell, d_forget_gate}, 1);

  torch::Tensor gates = d_sigmoid(gate_weights.view(d_gates.sizes()));

  d_gates *= gates;

  torch::Tensor d_weights = d_gates.t().mm(X);
  // sum across rows i.e. sum of columns,
  // keepdim=true means we're getting a result that has 1 row, columns same as before

  torch::Tensor d_bias = d_gates.sum(0, true); // not entirely sure why we're summing but I can see that the resulting shape is correct

  torch::Tensor d_X = d_gates.mm(weights);
  const int state_size = grad_h.size(1);
  torch::Tensor d_old_h = d_X.slice(1, 0, state_size);
  torch::Tensor d_input = d_X.slice(1, state_size);

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward pass (cuda)");
  m.def("backward", &sublstm_backward, "backward pass (cpp)");
}
