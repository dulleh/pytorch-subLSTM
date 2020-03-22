/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * and taking guidance from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda.cpp
  */
//Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
#include <torch/extension.h>
#include <iostream>
#include <cassert>
#include <chrono>
#include <fstream>
#include <cmath>
//namespace py = pybind11;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
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
  //std::ofstream outfile("cudaforwardtimes.csv", std::ios_base::app);

  CHECK_INPUT(input);
  CHECK_INPUT(weights);
  CHECK_INPUT(bias);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);
  auto start = std::chrono::high_resolution_clock::now();

  auto output = forward_cuda(input, weights, bias, old_h, old_cell);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
  //outfile << duration.count() / std::pow(10, 9) << ",";

  return output;
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
/**std::cout << "dE/dh" << grad_h.sizes() << std::endl; // 4,350
std::cout << "grad_cell" << grad_cell.sizes() << std::endl; // 4,350
std::cout << "new_cell" << new_cell.sizes() << std::endl; // 4,350
std::cout << "input_gate" << input_gate.sizes() << std::endl;// 4,350
std::cout << "output_gate" << output_gate.sizes() << std::endl;// 4,350
std::cout << "forget_gate" << forget_gate.sizes() << std::endl;// 4,350
std::cout << "candidate_cell" << candidate_cell.sizes() << std::endl;// 4,350
std::cout << "X" << X.sizes() << std::endl;// 4,352g
std::cout << "gate_weights" << gate_weights.sizes() << std::endl; // 4,4, 350
std::cout << "weights" << weights.sizes() << std::endl; // 1400, 352
std::cout << "old_cell" << old_cell.sizes() << std::endl; // 4, 350
**/

  auto output = backward_cuda(grad_h, grad_cell, new_cell, input_gate, output_gate, forget_gate, candidate_cell, X, gate_weights, weights, old_cell);
 return output;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward, "forward pass (cuda)");
  m.def("backward", &backward_sublstm, "backward pass (cuda)");
}
