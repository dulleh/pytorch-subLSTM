/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * Taking cues from  	https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda_kernel.cu
  */
#include <torch/all.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#include <stdio.h>

namespace {

	template <typename scalar_t>
	__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
	  return 1.0f / (1.0f + exp(-z));
	}

	template <typename scalar_t>
	__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
	  const auto s = sigmoid(z);
	  return (1.0f - s) * s;
	}

	template <typename scalar_t>
	__global__ void forward_cuda_kernel(
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		/**torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell
		**/
		torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> outputs
	)
	{
	  //batch index
	  const int n = blockIdx.y;
	  // column index ie output state index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < state_size){
			// input_gate
			const scalar_t ig = sigmoid(gates[n][0][c]);
			outputs[0][n][c] = ig;
			//input_gate[n][c] = ig;

			// output_gate
			const scalar_t og = sigmoid(gates[n][1][c]);
			outputs[1][n][c] = og;
			//output_gate[n][c] = og;

			// forget_gate
			const scalar_t fg = sigmoid(gates[n][3][c]);
			outputs[2][n][c] = fg;
			//forget_gate[n][c] = fg;

			// candidate_cell
			const scalar_t zg = sigmoid(gates[n][2][c]);
			outputs[3][n][c] = zg;
			//candidate_cell[n][c] = zg;

			// new_cell
			const scalar_t nc = (old_cell[n][c] * fg) + (zg - ig);
			//outputs[1][n][c] = nc;
			new_cell[n][c] = nc;

			// new_h
			//outputs[0][n][c] = sigmoid(nc) - og;
			new_h[n][c] = sigmoid(nc) - og;
		}
	}

	template <typename scalar_t>
	__global__ void new_forward_cuda_kernel(
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> outputs
	) {
		//batch index
		const int n = blockIdx.y;
		// column index ie output state index
		const int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c < state_size){
			const int fourC = 4 * c;
			// input_gate
			const scalar_t ig = sigmoid(gate_weights[n][fourC]);
			outputs[0][n][c] = ig;

			// output_gate
			const scalar_t og = sigmoid(gate_weights[n][fourC +1]);
			outputs[1][n][c] = og;

			// forget_gate
			const scalar_t fg = sigmoid(gate_weights[n][fourC +2]);
			outputs[2][n][c] = fg;

			// candidate_cell
			const scalar_t zg = sigmoid(gate_weights[n][fourC +3]);
			outputs[3][n][c] = zg;

			// new_cell
			const scalar_t nc = (old_cell[n][c] * fg) + (zg - ig);
			new_cell[n][c] = nc;

			// new_h
			new_h[n][c] = sigmoid(nc) - og;
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
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_gates,
		torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d_bias)
		{
	  // batch index
	  const int n = blockIdx.y;
	  // column index ie output state index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < state_size){
			// Calculate column indices
			const int igC = c;
			const int ogC = igC + state_size;
			const int zgC = ogC + state_size;
			const int fgC = zgC + state_size;

		  const scalar_t d_new_cell = (grad_h[n][c] * d_sigmoid(new_cell[n][c])) + grad_cell[n][c];
		  d_old_cell[n][c] = d_new_cell * forget_gate[n][c];
			// d_input_gate pre-activation
			const scalar_t dig = -d_new_cell * d_sigmoid(gate_weights[n][0][c]);
		  d_gates[n][igC] = dig;
			// d_output_gate  pre-activation
			const scalar_t dog = -grad_h[n][c] * d_sigmoid(gate_weights[n][1][c]);
		  d_gates[n][ogC] = dog;
			// d_candidate_cell pre-activation
			const scalar_t dzg = d_new_cell * d_sigmoid(gate_weights[n][2][c]);
		  d_gates[n][zgC] = dzg;
			// d_forget_gate pre-activation
			const scalar_t dfg = (d_new_cell * old_cell[n][c]) * d_sigmoid(gate_weights[n][3][c]);
		  d_gates[n][fgC] = dfg;

			// d_bias calculation
			// TODO: replace 4 atomicAdds with 1 some how?
			// 				- can't just switch block/thread direction as we lose memory contiguity
			// but actually are we even making use of memory contiguity if the columns are so far apart?
			atomicAdd(&d_bias[igC], dig);
			atomicAdd(&d_bias[ogC], dog);
			atomicAdd(&d_bias[zgC], dzg);
			atomicAdd(&d_bias[fgC], dfg);
	  }
	}

	template <typename scalar_t>
	__global__ void new_backward_cuda_kernel(
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_gates,
		torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d_bias) {
		// batch index
		const int n = blockIdx.y;
		// column index ie output state index
		const int c = blockIdx.x * blockDim.x + threadIdx.x;
		if (c < state_size){
			// Calculate column indices
			const int fourC = 4*c;
			const int igC = fourC;
			const int ogC = fourC + 1;
			const int fgC = fourC + 2;
			const int zgC = fourC + 3;

			const scalar_t d_new_cell = (grad_h[n][c] * d_sigmoid(new_cell[n][c])) + grad_cell[n][c];
			d_old_cell[n][c] = d_new_cell * forget_gate[n][c];
			// d_input_gate pre-activation
			const scalar_t dig = -d_new_cell * d_sigmoid(gate_weights[n][igC]);
			d_gates[n][igC] = dig;
			// d_output_gate  pre-activation
			const scalar_t dog = -grad_h[n][c] * d_sigmoid(gate_weights[n][ogC]);
			d_gates[n][ogC] = dog;
			// d_forget_gate pre-activation
			const scalar_t dfg = (d_new_cell * old_cell[n][c]) * d_sigmoid(gate_weights[n][fgC]);
			d_gates[n][fgC] = dfg;
			// d_candidate_cell pre-activation
			const scalar_t dzg = d_new_cell * d_sigmoid(gate_weights[n][zgC]);
			d_gates[n][zgC] = dzg;

			// d_bias calculation
			// TODO: replace 4 atomicAdds with 1 some how?
			// 				- can't just switch block/thread direction as we lose memory contiguity
			// but actually are we even making use of memory contiguity if the columns are so far apart?
			atomicAdd(&d_bias[igC], dig);
			atomicAdd(&d_bias[ogC], dog);
			atomicAdd(&d_bias[zgC], dzg);
			atomicAdd(&d_bias[fgC], dfg);
		}
	}

}

std::vector<torch::Tensor> forward_cuda(
    torch::Tensor input,
    torch::Tensor gate_weights,
    torch::Tensor old_h,
    torch::Tensor old_cell)
{
	auto options = gate_weights.options().requires_grad(true);

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

	//TODO: manually index so that you can try doing it with contiguous memory accesses
	// This is just for convenience in indexing. If you pull it up into
	// Python you might have to call .contiguous
	/////auto gates = gate_weights.view({batch_size, 4, state_size});

	auto intermediates = torch::empty({4, batch_size, state_size}, options);

	auto new_h = torch::empty_like(old_cell);
  auto new_cell = torch::empty_like(old_cell);
/**  auto input_gate = torch::empty_like(old_cell);
  auto output_gate = torch::empty_like(old_cell);
  auto forget_gate = torch::empty_like(old_cell);
  auto candidate_cell = torch::empty_like(old_cell);
**/

  /**
    * As for the kernel launch itself, we are here specifying that each CUDA block will have 1024 threads, and that the
    * entire GPU grid is split into as many blocks of 1 x 1024 threads as are required to fill our matrices with one
    * thread per component. For example, if our state size was 2048 and our batch size 4, we’d launch a total of
    * 4 x 2 = 8 blocks with each 1024 threads.=
    * Source: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
    **/
  const int threads = 1024;
  const dim3 blocks((state_size + threads - 1) / threads, batch_size);

/**
  AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", ([&] {
    forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
			batch_size,
			state_size,
	    gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
			old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			intermediates.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
		);
  }));
**/
	AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", ([&] {
		new_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
			batch_size,
			state_size,
			gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			intermediates.packed_accessor32<scalar_t, 3, torch::RestrictPtrTraits>()
		);
	}));

	return {new_h, new_cell, intermediates};
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
	// Store no gradients
	torch::NoGradGuard no_grad;
	torch::Device device(torch::kCUDA, 0);
	auto options = grad_h.options().requires_grad(false);

  const auto batch_size = grad_h.size(0);
  const auto state_size = grad_h.size(1);

	//auto gates = gate_weights.view({batch_size, 4, state_size});

	// auto d_new_cell  -- Don't need this as it is not returned, and used only within the kernel
	auto d_old_cell = torch::empty({old_cell.size(0), old_cell.size(1)}, options);
	auto d_gates = torch::empty({batch_size, 4*state_size}, options);
	auto d_bias = torch::zeros({4*state_size}, options);

	const int threads = 1024;
	const dim3 blocks((state_size + threads - 1) / threads, batch_size);

/**
  AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
    backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
      batch_size,
      state_size,
			grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
      new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			forget_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			gates.packed_accessor32<scalar_t,3,torch::RestrictPtrTraits>(),
			old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_gates.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
		);
	}));
**/
	AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
		new_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
			batch_size,
			state_size,
			grad_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			grad_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			forget_gate.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_gates.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
		);
	}));



	torch::Tensor d_weights = d_gates.t().contiguous().mm(X);
	torch::Tensor d_X = d_gates.mm(weights);

	torch::Tensor d_old_h = d_X.slice(1, 0, state_size); // first state_size columns
	torch::Tensor d_input = d_X.slice(1, state_size); // from column [state_size + 1] to the end

	return {d_old_h, d_input, d_weights, d_bias, d_old_cell, d_gates};
}
