/**
 * Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
 */
#include <torch/extension.h>

//namespace py = pybind11;

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
    auto s = torch::sigmoid(z);
    return (1 - s) * s;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d_sigmoid", &lltm_forward, "sigmoid differential");
}