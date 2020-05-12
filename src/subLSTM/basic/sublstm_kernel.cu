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
		const torch::PackedTensorAccessor<scalar_t,3,torch::RestrictPtrTraits,size_t> gates,
		const torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> old_cell,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_h,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> new_cell,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> input_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> output_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> forget_gate,
		torch::PackedTensorAccessor<scalar_t,2,torch::RestrictPtrTraits,size_t> candidate_cell) {
	  //batch index
	  const int n = blockIdx.y;
	  // column index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < gates.size(2)){
		input_gate[n][c] = sigmoid(gates[n][0][c]);
		output_gate[n][c] = sigmoid(gates[n][1][c]);
		candidate_cell[n][c] = sigmoid(gates[n][2][c]);
		forget_gate[n][c] = sigmoid(gates[n][3][c]);
		new_cell[n][c] =
			(old_cell[n][c] * forget_gate[n][c]) + (candidate_cell[n][c] - input_gate[n][c]);
		new_h[n][c] = sigmoid(new_cell[n][c]) - output_gate[n][c];
	  }
	}
	
	template <typename scalar_t>
	__global__ void backward_cuda_kernel(
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
		const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_gates) {
	  // batch index
	  const int n = blockIdx.y;
	  // column index ie output state index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < state_size){
		  const auto d_new_cell = (grad_h[n][c] * d_sigmoid(new_cell[n][c])) + grad_cell[n][c];
		  d_old_cell[n][c] = d_new_cell * forget_gate[n][c];
		  d_gates[n][c] = -d_new_cell * d_sigmoid(gate_weights[n][0][c]); // d_input_gate pre-activation
		  d_gates[n][state_size + c] = -grad_h[n][c] * d_sigmoid(gate_weights[n][1][c]); // d_output_gate  pre-activation
		  d_gates[n][2*state_size + c] = d_new_cell * d_sigmoid(gate_weights[n][2][c]); // d_candidate_cell pre-activation
		  d_gates[n][3*state_size + c] = (d_new_cell * old_cell[n][c]) * d_sigmoid(gate_weights[n][3][c]); // d_forget_gate pre-activation
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
  auto gate_weights = torch::addmm(bias, X, weights.transpose(0, 1));

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

  auto gates = gate_weights.view({batch_size, 4, state_size});
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
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

  AT_DISPATCH_FLOATING_TYPES(gates.type(), "forward_cuda", ([&] {
    forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
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
    torch::Tensor old_cell) {
    const auto batch_size = grad_h.size(0);
    const auto state_size = grad_h.size(1);

	// auto d_new_cell  -- Don't need this as it is not returned, and used only within the kernel
	auto d_old_cell = torch::zeros_like(old_cell);
	auto d_gates = torch::zeros({batch_size, 4*state_size}, weights.options());

	const int threads = 1024;
    const dim3 blocks((state_size + threads - 1) / threads, batch_size);

    AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
      backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        batch_size,
        state_size,
		grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
        new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
		grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
		forget_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
		gate_weights.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
		old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
		d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
		d_gates.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>());
    }));

	//std::cout << "cu: d_gates[0][12]" << d_gates[1][1] << std::endl;
	//std::cout << "cu: d_old_cell[1][300]" << d_old_cell[1][300] << std::endl;

	torch::Tensor d_weights = d_gates.t().mm(X);

	// sum across rows i.e. sum of columns,
	// keepdim=true means we're getting a result that has 1 row, columns same as before
	torch::Tensor d_bias = d_gates.sum(0, true); // not entirely sure why we're summing but I can see that the resulting shape is correct

	torch::Tensor d_X = d_gates.mm(weights);
	torch::Tensor d_old_h = d_X.slice(1, 0, state_size); // first state_size columns
	torch::Tensor d_input = d_X.slice(1, state_size); // from column [state_size + 1] to the end

	return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}