/**
  * This is all from https://pytorch.org/tutorials/advanced/cpp_extension.html
  * Taking cues from  	https://github.com/pytorch/extension-cpp/blob/master/cuda/lltm_cuda_kernel.cu
  */
#include <torch/all.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAMultiStreamGuard.h>
#include <ATen/cuda/CUDAEvent.h>

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

// Initial source code from
//	http://ecee.colorado.edu/~siewerts/extra/code/example_code_archive/a490dmis_code/CUDA/cuda_work/samples/0_Simple/matrixMul/matrixMul.cu
// Modified to include a vector addition, index tensors and cope with arbitrary matrix sizes
// Does C = AB + beta with A,B,C rectangular matrices of appr. sizes, and beta a vector
template <typename scalar_t, int BLOCK_SIZE>
__global__ void mmAdd(
	torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
	const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
	const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> B,
	const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> beta
	)
	{
	  // Block index:
		// BLOCK_SIZE*bx gives first column in this block of C
		// BLOCK_SIZE*by gives first row in this block of C
	  const int bx = blockIdx.x;
	  const int by = blockIdx.y;

	  // Thread index
	  int tRow = threadIdx.y;
	  int tCol = threadIdx.x;

	  // Index of the first element of the first sub-matrix of A
		// processed by the block
	  //int aBegin = wA * BLOCK_SIZE * by;
		int aBeginRow = by * BLOCK_SIZE;
		int aBeginCol = 0;

	  // Index of the first sub-matrix of B processed by the block
		//int bBegin = BLOCK_SIZE * bx;
		int bBeginRow = 0;
	  int bBeginCol = bx * BLOCK_SIZE;

		int n = aBeginRow + tRow;
		int c = bBeginCol + tCol;

	  // Accumulated output for the C value at
		// C[by * BLOCK_SIZE + tRow][bx * BLOCK_SIZE + tCol]
	  scalar_t Csub = 0;

 	  // we save the last potential block as a special case rather than have it
		// result in an extra condition every loop
		while (aBeginCol < A.size(1))
		{
			// Declaration of the shared memory arrays
      __shared__ scalar_t As[BLOCK_SIZE][BLOCK_SIZE];
      __shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

      // Load the matrices from global to shared memory;
			// each thread loads one element of each matrix
			// NOTE: These do not correspond to the actual result indices
      //As[ty][tx] = A[a + wA * ty + tx];
      //Bs[ty][tx] = B[b + wB * ty + tx];
			int aRow = aBeginRow + tRow;
			int aCol = aBeginCol + tCol;
			As[tRow][tCol] = (aRow < A.size(0) && aCol < A.size(1)) ? A[aRow][aCol] : scalar_t(); // 0
			int bRow = bBeginRow + tRow;
			int bCol = bBeginCol + tCol;
			Bs[tRow][tCol] = (bRow < B.size(0) && bCol < B.size(1)) ? B[bRow][bCol] : scalar_t();

      // Synchronize to make sure the matrices are loaded
      __syncthreads();

      // Multiply the two sub-matrices together
	#pragma unroll
      for (int k = 0; k < BLOCK_SIZE; ++k)
      {
          Csub += As[tRow][k] * Bs[k][tCol];
      }

			aBeginCol += BLOCK_SIZE;
			bBeginRow += BLOCK_SIZE;
      // Synchronize to make sure that the preceding
      // computation is done before loading two new
      // sub-matrices of A and B in the next iteration
      __syncthreads();
	  }

		if (n < C.size(0) && c < C.size(1)) {
		  // Write the block sub-matrix to global mem. each thread writes one element
			// beta should already get cached?
		  //int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
		  //C[c + wB * ty + tx] = Csub;
			C[n][c] = Csub + beta[c];
		}
	}

	template <typename scalar_t>
	__global__ void forward_cuda_kernel
	(
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		/**torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell
		**/
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate
	)
	{
	  //batch index
	  const int n = blockIdx.y;
	  // column index ie output state index
	  const int c = blockIdx.x * blockDim.x + threadIdx.x;
	  if (c < state_size){
			// input_gate
			const scalar_t ig = sigmoid(gate_weights[n][c]);

			// output_gate
			const scalar_t og = sigmoid(gate_weights[n][c + state_size]);

			// forget_gate
			const scalar_t fg = sigmoid(gate_weights[n][c + 2*state_size]);
			forget_gate[n][c] = fg;

			// candidate_cell
			const scalar_t zg = sigmoid(gate_weights[n][c + 3*state_size]);

			// new_cell
			const scalar_t nc = (old_cell[n][c] * fg) + (zg - ig);
			new_cell[n][c] = nc;

			// new_h
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
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate
	)
	{
		const int output_index = blockIdx.x * blockDim.x + threadIdx.x;

		const int n = output_index / state_size;
		const int c = output_index % state_size;

		if (n < batch_size && c < state_size){
			const int fourC = 4 * c;
			// input_gate
			const scalar_t ig = sigmoid(gate_weights[n][fourC]);

			// output_gate
			const scalar_t og = sigmoid(gate_weights[n][fourC +1]);

			// forget_gate
			const scalar_t fg = sigmoid(gate_weights[n][fourC +2]);
			forget_gate[n][c] = fg;

			// candidate_cell
			const scalar_t zg = sigmoid(gate_weights[n][fourC +3]);

			// new_cell
			const scalar_t nc = (old_cell[n][c] * fg) + (zg - ig);
			new_cell[n][c] = nc;

			// new_h
			new_h[n][c] = sigmoid(nc) - og;
		}
	}

	template <typename scalar_t>
	__global__ void backward_cuda_kernel
	(
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
		torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d_bias)
		{
		const int output_index = blockIdx.x * blockDim.x + threadIdx.x;

		const int n = output_index / state_size;
		const int c = output_index % state_size;

		if (n < batch_size && c < state_size) {
			// Calculate column indices
			const int igC = 4*c;
			const int ogC = igC + 1;
			const int fgC = igC + 2;
			const int zgC = igC + 3;

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
			atomicAdd(&d_bias[igC], dig);
			atomicAdd(&d_bias[ogC], dog);
			atomicAdd(&d_bias[zgC], dzg);
			atomicAdd(&d_bias[fgC], dfg);
		}
	}

}

std::vector<torch::Tensor> forward_cuda(
	torch::Tensor input,
	torch::Tensor old_h,
	torch::Tensor old_cell,
	torch::Tensor weightsT,
	torch::Tensor bias
)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

	auto options = weightsT.options().requires_grad(false);

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

	auto X = torch::cat({old_h, input}, 1);

	auto gate_weights = torch::empty({X.size(0), weightsT.size(1)}, options);
	int t = 16;
	dim3 THREADS(t, t);
  dim3 GRID((gate_weights.size(1) + t - 1) / t,
						(gate_weights.size(0) + t - 1) / t);
	AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "custom_mmadd", ([&] {
		mmAdd<scalar_t, 16><<<GRID, THREADS, 0, stream>>>(
			gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			weightsT.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
		);
	}));

	auto new_h = torch::empty({batch_size, state_size}, options);
  auto new_cell = torch::empty({batch_size, state_size}, options);
	auto forget_gate = torch::empty({batch_size, state_size}, options);

/**
  * As for the kernel launch itself, we are here specifying that each CUDA block will have 1024 threads, and that the
  * entire GPU grid is split into as many blocks of 1 x 1024 threads as are required to fill our matrices with one
  * thread per component. For example, if our state size was 2048 and our batch size 4, we’d launch a total of
  * 4 x 2 = 8 blocks with each 1024 threads.=
  * Source: https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
  **/

	const int threads = 256;
	const dim3 blocks((state_size*batch_size + threads - 1) / threads, 1);

	AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", ([&] {
		new_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
			batch_size,
			state_size,
			gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			forget_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
		);
	}));

	return {new_h, new_cell, forget_gate, gate_weights, X};
}

std::vector<torch::Tensor> backward_cuda(
		torch::Tensor grad_h,
		torch::Tensor grad_cell,
		torch::Tensor new_cell,
		torch::Tensor forget_gate,
		torch::Tensor X,
		torch::Tensor gate_weights, // gate outputs, pre-activation
		torch::Tensor weights, // actual weights in the gates
		torch::Tensor old_cell)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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

	const int threads = 256;
	const dim3 blocks((state_size*batch_size + threads - 1) / threads, 1);

	AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
		new_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
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

	torch::Tensor d_weights = X.t().mm(d_gates);
	torch::Tensor d_X = d_gates.mm(weights.t());

	torch::Tensor d_old_h = d_X.slice(1, 0, state_size); // first state_size columns
	torch::Tensor d_input = d_X.slice(1, state_size); // from column [state_size + 1] to the end

	return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
