/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * and taking guidance from https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda.cpp
  */
//Includes ATen (tensor library), pybind11, and headers to manage the interactions between the two.
#include <torch/all.h>
#include <torch/python.h>
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
    torch::Tensor gate_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell);

std::vector<torch::Tensor> forward_sublstm(
    torch::Tensor input,
    torch::Tensor gate_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
/**
  CHECK_INPUT(input);
  CHECK_INPUT(gate_weights);
  CHECK_INPUT(old_h);
  CHECK_INPUT(old_cell);
**/
  return forward_cuda(input, gate_weights, old_h, old_cell);
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
/**  CHECK_INPUT(grad_h);
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
**/
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
  return
    backward_cuda(grad_h, grad_cell, new_cell, input_gate, output_gate, forget_gate,
            candidate_cell, X, gate_weights, weights, old_cell);

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

class SubLSTMFunction: public torch::autograd::Function<SubLSTMFunction> {
 public:
  static torch::autograd::tensor_list forward(
      torch::autograd::AutogradContext* ctx,
      torch::Tensor input,
      torch::Tensor weights,
      torch::Tensor bias,
      torch::Tensor old_h,
      torch::Tensor old_cell,
      std::vector<torch::Tensor> non_vars ) {
    auto X = non_vars[0];
    auto gate_weights = non_vars[1];

    auto output_list = forward_sublstm(input, gate_weights, old_h, old_cell);

    auto new_h = output_list[0];
    auto new_cell = output_list[1];

    auto stacked_intermediates = output_list[2];
  	auto intermediates = stacked_intermediates.unbind();
    auto input_gate = intermediates[0];
    auto output_gate = intermediates[1];
  	auto forget_gate = intermediates[2];
    auto candidate_cell = intermediates[3];

    ctx->save_for_backward({new_cell, input_gate, output_gate, forget_gate, candidate_cell, weights, X, gate_weights, old_h, old_cell});

    return {new_h, new_cell};
  }

  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grads)  {
        auto grad_h    = grads[0];
        auto grad_cell = grads[1];

        auto saved_vars = ctx->get_saved_variables();

        auto X = saved_vars[6];
        auto gate_weights = saved_vars[7];
        auto old_h = saved_vars[8];
        auto old_cell = saved_vars[8];

        auto outputs = backward_sublstm(
          grad_h, grad_cell, saved_vars[0], saved_vars[1],
          saved_vars[2], saved_vars[3], saved_vars[4],
          X, gate_weights, saved_vars[5], old_cell );

        auto d_old_h = outputs[0];
        auto d_input = outputs[1];
        auto d_weights = outputs[2];
        auto d_bias = outputs[3];
        auto d_old_cell = outputs[4];
        auto d_gates = outputs[5];

        return {d_input, d_weights, d_bias, d_old_h, d_old_cell, torch::Tensor()};
  }
};

torch::autograd::tensor_list sublstm_apply(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell,
    std::vector<torch::Tensor> non_vars // = [X, gate_weights]
  ) {
  return SubLSTMFunction::apply(input, weights, bias, old_h, old_cell, non_vars);
}

static auto registry =
  torch::RegisterOperators("sublstm::forward", &forward_sublstm)
        .op("sublstm::backward", &backward_sublstm)
        .op("sublstm::apply", &sublstm_apply);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &forward_sublstm, "forward pass (cuda)");
  m.def("backward", &backward_sublstm, "backward pass (cuda)");
}
