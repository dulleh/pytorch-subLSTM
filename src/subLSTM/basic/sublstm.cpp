/**
 * Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
 */
#include <torch/extension.h>

//namespace py = pybind11;

#include <iostream>

// CUDA forward declarations
torch::Tensor d_sigmoid_cuda(torch::Tensor z);

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor d_sigmoid(torch::Tensor z) {
    CHECK_INPUT(z);
    return d_sigmoid_cuda(z);
    //auto s = torch::sigmoid(z);
    //return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d_sigmoid", &d_sigmoid, "sigmoid differential (cuda)");
}