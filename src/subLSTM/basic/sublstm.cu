/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  */
#include "sublstm.cuh"

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
  const auto s = sigmoid(z);
  return (1.0 - s) * s;
}

template <typename scalar_t>
__global__ void forward_cuda_kernel(
    //TODO: I changed this 3->4 because we needed a forget gate?
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> gates,
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
  //batch index
  const int n = blockIdx.y;
  // column index
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < gates.size(2)){
  //TODO: We need a forget gate, but also need to check if these are ordered correctly
    input_gate[n][c] = sigmoid(gates[n][0][c]);
    output_gate[n][c] = sigmoid(gates[n][1][c]);
    candidate_cell[n][c] = sigmoid(gates[n][2][c]);
    forget_gate[n][c] = sigmoid(gates[n][3][c]);
    new_cell[n][c] =
        old_cell[n][c] * forget_gate + candidate_cell[n][c] - input_gate[n][c];
    new_h[n][c] = sigmoid(new_cell[n][c]) - output_gate[n][c];
  }
}