/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * and taking guidance from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda.cpp
  */
//Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
#include <torch/extension.h>
#include <torch/script.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <cmath>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

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


std::vector<torch::Tensor> backward_cuda(
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
    torch::Tensor old_cell);

std::vector<torch::Tensor> backward_sublstm(
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
// batch-size=4, nhid=450,
/**
std::cout << "dE/dh" << grad_h.sizes() << std::endl; // 4,350
std::cout << "grad_cell" << grad_cell.sizes() << std::endl; // 4,350
std::cout << "new_cell" << new_cell.sizes() << std::endl; // 4,350
std::cout << "input_gate" << input_gate.sizes() << std::endl;// 4,350
std::cout << "output_gate" << output_gate.sizes() << std::endl;// 4,350
std::cout << "forget_gate" << forget_gate.sizes() << std::endl;// 4,350
std::cout << "candidate_cell" << candidate_cell.sizes() << std::endl;// 4,350
std::cout << "X" << X.sizes() << std::endl;// 4,352
std::cout << "gate_weights" << gate_weights.sizes() << std::endl; // 4,4, 350
std::cout << "weights" << weights.sizes() << std::endl; // 1400, 352
std::cout << "old_cell" << old_cell.sizes() << std::endl; // 4, 350
**/
  auto output = backward_cuda(grad_h, grad_cell, new_cell, input_gate, output_gate, forget_gate, candidate_cell, X, gate_weights, weights, old_cell);


 return output;
/**
  torch::Tensor d_new_cell = (grad_h * d_sigmoid(new_cell)) + (grad_cell);

  torch::Tensor d_old_cell = d_new_cell * forget_gate; // dE/dct-1 = dE/dct * dct/dct-1 = delta(ct) * ft

  //std::cout << "d_old_cell" << d_old_cell.sizes() << std::endl; // 4, 350

  torch::Tensor d_input_gate = -d_new_cell; // this is delta(it)
  torch::Tensor d_output_gate = -grad_h; // ht = sigmoid(ct) - ot (where ot is post activation)
  torch::Tensor d_candidate_cell = d_new_cell; // this is delta(zt)
  torch::Tensor d_forget_gate = d_new_cell * old_cell;

  torch::Tensor d_gates =
        torch::cat({d_input_gate, d_output_gate, d_candidate_cell, d_forget_gate}, 1);

  //std::cout << "d_gates" << d_gates.sizes() << std::endl; // 4, 1400


  torch::Tensor d_sigm_gates = d_sigmoid(gate_weights.view(d_gates.sizes()));

  d_gates *= d_sigm_gates;

  //std::cout << "c+: d_gates[0][12]" << d_gates[1][1] << std::endl;
  //std::cout << "c+: d_old_cell[1][300]" << d_old_cell[1][300] << std::endl;



  torch::Tensor d_weights = d_gates.t().mm(X);

  // sum across rows i.e. sum of columns,
  // keepdim=true means we're getting a result that has 1 row, columns same as before
  torch::Tensor d_bias = d_gates.sum(0, true); // not entirely sure why we're summing but I can see that the resulting shape is correct

  torch::Tensor d_X = d_gates.mm(weights);
  const int state_size = grad_h.size(1);
  torch::Tensor d_old_h = d_X.slice(1, 0, state_size); // first state_size columns
  torch::Tensor d_input = d_X.slice(1, state_size); // from column [state_size + 1] to the end

  return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};

**/
}

static auto registry =
  torch::RegisterOperators("sublstm::forward", &forward)
        .op("sublstm::backward", &backward_sublstm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward pass (cuda)");
  m.def("backward", &backward_sublstm, "backward pass (cuda)");
}
