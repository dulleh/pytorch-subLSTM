/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * Taking cues from  	https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda_kernel.cu
  */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <stdio.h>

namespace {

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
		const int X_size,
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> X,
		const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> weights,
		const torch::PackedTensorAccessor<scalar_t,1,torch::RestrictPtrTraits,size_t> bias,
		torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
		const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> forget_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell) {
	  //batch index
	  const int n = blockIdx.y;
	  // column index ie output state index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < state_size){
	  	for (int k = 0; k < X_size; k++) {
		  gates[n][0][c] += X[n][k] * weights[c][k];
	      gates[n][1][c] += X[n][k] * weights[state_size + c][k];
		  gates[n][2][c] += X[n][k] * weights[2*state_size + c][k];
		  gates[n][3][c] += X[n][k] * weights[3*state_size + c][k];
		}
		gates[n][0][c] += bias[c];
		gates[n][1][c] += bias[state_size + c];
		gates[n][2][c] += bias[2*state_size + c];
		gates[n][3][c] += bias[3*state_size + c];

		input_gate[n][c] = sigmoid(gates[n][0][c]);
		output_gate[n][c] = sigmoid(gates[n][1][c]);
		candidate_cell[n][c] = sigmoid(gates[n][2][c]);
		forget_gate[n][c] = sigmoid(gates[n][3][c]);
		new_cell[n][c] =
			(old_cell[n][c] * forget_gate[n][c]) + (candidate_cell[n][c] - input_gate[n][c]);
		new_h[n][c] = sigmoid(new_cell[n][c]) - output_gate[n][c];
	  }
	}

}

std::vector<torch::Tensor> forward_cuda(
    torch::Tensor input,
    torch::Tensor weights,
    torch::Tensor bias,
    torch::Tensor old_h,
    torch::Tensor old_cell) {
  auto X = torch::cat({old_h, input}, /*dim=*/1);

  //std::cout << "X: " << X << std::endl;
  //std::cout << "weights: " << weights << std::endl;
 // std::cout << "bias: " << bias << std::endl;

  const auto X_size = X.size(1);
  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

//  std::cout << "X_size: " << X_size << std::endl;
//  std::cout << "batch_size: " << batch_size << std::endl;
//  std::cout << "state_size: " << state_size << std::endl;

  auto gates = torch::zeros({batch_size, 4, state_size}, weights.options());
  auto new_h = torch::zeros_like(old_cell);
  auto new_cell = torch::zeros_like(old_cell);
  auto input_gate = torch::zeros_like(old_cell);
  auto output_gate = torch::zeros_like(old_cell);
  auto forget_gate = torch::zeros_like(old_cell);
  auto candidate_cell = torch::zeros_like(old_cell);

  /**
    * As for the kernel launch itself, we are here specifying that each CUDA block will have 1024 threads, and that the
    * entire GPU grid is split into as many blocks of 1 x 1024 threads as are required to fill our matrices with one
    * thread per component. For example, if our state size was 2048 and our batch size 4, we’d launch a total of
    * 4 x 2 = 8 blocks with each 1024 threads.=
    * Source: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
    **/
  const int threads = 512;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(input.type(), "forward_cuda", ([&] {
    forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
		X_size,
		batch_size,
		state_size,
		X.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        weights.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
		bias.packed_accessor<scalar_t,1,torch::RestrictPtrTraits,size_t>(),
	    gates.packed_accessor<scalar_t,3,torch::RestrictPtrTraits,size_t>(),
		old_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_h.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        new_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        input_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        output_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        forget_gate.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>(),
        candidate_cell.packed_accessor<scalar_t,2,torch::RestrictPtrTraits,size_t>());
  }));

  return {new_h, new_cell, input_gate, output_gate, forget_gate, candidate_cell, X, gates};
}
