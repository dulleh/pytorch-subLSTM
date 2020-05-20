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

namespace
{

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
	__global__ void mmAdd
	(
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

	/**
		* Based on:
	  *  https://pytorch.org/tutorials/advanced/cpp_extension.html#writing-a-mixed-c-cuda-extension
	  **/
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
	struct scalar_t4 {
	  scalar_t input;
		scalar_t output;
		scalar_t forget;
		scalar_t candidate;
	};

	/**
	  * All tensors must be contiguous and row-major.
		*/
	template <typename scalar_t>
	__global__ void new_forward_cuda_kernel
	(
		const int size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate
	)
	{
		const int output_index = blockIdx.x * blockDim.x + threadIdx.x;

	 // output indices
	 // int n = output_index / state_size;
	 // int c = output_index % state_size;

		if (output_index < size){
			 // vectorized load
			scalar_t4<scalar_t> pre_activation = reinterpret_cast<scalar_t4<scalar_t>*>(gate_weights.data())[output_index];
			// load the only other value required
			scalar_t old_cell_val = old_cell.data()[output_index];

			// apply gate activations
			scalar_t ig = sigmoid(pre_activation.input);
			scalar_t og = sigmoid(pre_activation.output);
			scalar_t fg = sigmoid(pre_activation.forget);
			scalar_t zg = sigmoid(pre_activation.candidate);
			// new_cell calculation
		 	scalar_t nc = (old_cell_val * fg) + (zg - ig);

			// store outputs into global
			forget_gate.data()[output_index] = fg;
			new_cell.data()[output_index] = nc;
			new_h.data()[output_index] = sigmoid(nc) - og;
		}
	}

	// threads per block is required to be BLOCK_SIZE^2
	template <typename scalar_t, int BLOCK_SIZE, int THREADS_PER_BLOCK>
	__global__ void fused_forward_cuda_kernel
	(
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> B,
		const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> beta,
		const int batch_size,
		const int state_size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate
	)
	{
	/**
	** mmAdd with additonal shared storage for outputs
	**/
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

		// Declaration of the shared memory arrays
		__shared__ scalar_t As[THREADS_PER_BLOCK];
		__shared__ scalar_t Bs[BLOCK_SIZE][BLOCK_SIZE];

		while (aBeginCol < A.size(1)) {

			// Load the matrices from global to shared memory;
			// each thread loads one element of each matrix
			// NOTE: These do not correspond to the actual result indices
			//As[ty][tx] = A[a + wA * ty + tx];
			//Bs[ty][tx] = B[b + wB * ty + tx];
			int aRow = aBeginRow + tRow;
			int aCol = aBeginCol + tCol;
			As[tRow*BLOCK_SIZE + tCol] = (aRow < A.size(0) && aCol < A.size(1)) ? A[aRow][aCol] : scalar_t(); // 0
			int bRow = bBeginRow + tRow;
			int bCol = bBeginCol + tCol;
			Bs[tRow][tCol] = (bRow < B.size(0) && bCol < B.size(1)) ? B[bRow][bCol] : scalar_t();

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Multiply the two sub-matrices together
	#pragma unroll
			for (int k = 0; k < BLOCK_SIZE; ++k)
			{
					Csub += As[tRow*BLOCK_SIZE + k] * Bs[k][tCol];
			}

			aBeginCol += BLOCK_SIZE;
			bBeginRow += BLOCK_SIZE;
			// Synchronize to make sure that the preceding
			// computation is done before loading two new
			// sub-matrices of A and B in the next iteration
			__syncthreads();
		}

		// assumes block are square with width BLOCK_SIZE

		if (n < C.size(0) && c < C.size(1)) {
			// Repurpose As to be used as a shared memory to collate for next kernel
			// don't move beta from here as it benefits a lot from broadcasting
			As[tRow*BLOCK_SIZE + tCol] = Csub  + beta[c];
		}

	// sync threads to make sure all activation values calculated
	__syncthreads();

	// reduce by 4 threads
	if (tCol % 4 == 0) {
		// notice we go to (c/4)th column
		int output_index = n*state_size + (c/4);

		/**
		 * new_forward_cuda_kernel
		 **/
		//if (n < batch_size && output_c < state_size){
		if (output_index < batch_size * state_size) {
			// vectorized loads
			scalar_t4<scalar_t> pre_activation = reinterpret_cast<scalar_t4<scalar_t>*>(As)[(tRow*BLOCK_SIZE + tCol) / 4];
			// load the only other value required
			scalar_t old_cell_val = old_cell.data()[output_index];

			// apply gate activations
			scalar_t ig = sigmoid(pre_activation.input);
			scalar_t og = sigmoid(pre_activation.output);
			scalar_t fg = sigmoid(pre_activation.forget);
			scalar_t zg = sigmoid(pre_activation.candidate);

			// new_cell calculation
		 	scalar_t nc = (old_cell_val * fg) + (zg - ig);

			// store outputs into global
			forget_gate.data()[output_index] = fg;
			new_cell.data()[output_index] = nc;
			new_h.data()[output_index] = sigmoid(nc) - og;

			// vectorized store
			reinterpret_cast<scalar_t4<scalar_t>*>(C.data())[output_index] = pre_activation;
		}
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
			const int fgC = ogC + state_size;
			const int zgC = fgC + state_size;

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
			atomicAdd(&d_bias[fgC], dfg);
			atomicAdd(&d_bias[zgC], dzg);
	  }
	}

	template <typename scalar_t>
	__global__ void new_backward_cuda_kernel(
		const int batch_size,
		const int state_size,
		const int size,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_h,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> grad_cell,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> forget_gate,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> gate_weights,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_old_cell,
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> d_gates,
		torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d_bias
	)
	{
		const int output_index = blockIdx.x * blockDim.x + threadIdx.x;

		const int n = output_index / state_size;
		const int c = output_index % state_size;

		//if (n < batch_size && c < state_size) {
		if (output_index < size) {
			// vectorized load
			scalar_t4<scalar_t> gate_vals = reinterpret_cast<scalar_t4<scalar_t>*>(gate_weights.data())[output_index];
			// load other values too
			scalar_t grad_h_val = grad_h[n][c];
			scalar_t grad_cell_val = grad_cell[n][c];
			scalar_t new_cell_val = new_cell[n][c];
			scalar_t old_cell_val = old_cell[n][c];
			scalar_t forget_gate_val = forget_gate[n][c];

			scalar_t d_new_cell = (grad_h_val * d_sigmoid(new_cell_val)) + grad_cell_val;
			// d_input_gate pre-activation
			gate_vals.input = -d_new_cell * d_sigmoid(gate_vals.input);
			// d_output_gate pre-activation
			gate_vals.output = -grad_h_val * d_sigmoid(gate_vals.output);
			// d_forget_gate pre-activation
			gate_vals.forget = (d_new_cell * old_cell_val) * d_sigmoid(gate_vals.forget);
			// d_candidate_cell pre-activation
			gate_vals.candidate = d_new_cell * d_sigmoid(gate_vals.candidate);

			// write outputs to global
			d_old_cell[n][c] = d_new_cell * forget_gate_val;
			// vectorized store to d_gates
			reinterpret_cast<scalar_t4<scalar_t>*>(d_gates.data())[output_index] = gate_vals;
			// d_bias calculation and store
			// TODO: replace 4 atomicAdds with 1 some how?
			atomicAdd(&d_bias[4*c], gate_vals.input);
			atomicAdd(&d_bias[4*c + 1], gate_vals.output);
			atomicAdd(&d_bias[4*c + 2], gate_vals.forget);
			atomicAdd(&d_bias[4*c + 3], gate_vals.candidate);
		}
	}


	// BLOCK_SIZEX is the number of COLUMNS in a block
	// BLOCK_SIZEY is the number of ROWS in a block
	// both are required to be the same as block dimensions
	// (we just need them at compile time)
	template <typename scalar_t, int BLOCK_SIZEX, int BLOCK_SIZEY, int THREADS_PER_BLOCK>
	__global__ void fused_backward_cuda_kernel(
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
		torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> d_bias,
		const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A, // X.t() NOT CONTIGUOUS SO DO NOT USE .data()
		// B is calculated in the kernel - it is d_gates
		torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C // d_weights
	)
	{
		// Block index:
		// BLOCK_SIZEX*bx gives first column in this block of C
		// BLOCK_SIZEY*by gives first row in this block of C
		const int bx = blockIdx.x;
		const int by = blockIdx.y;

		// Thread index
		const int tCol = threadIdx.x;
		const int tRow = threadIdx.y;

		//__shared__ scalar_t Bs[BLOCK_SIZEY][BLOCK_SIZEX];
		__shared__ scalar_t Bs[THREADS_PER_BLOCK];

		int c = bx * BLOCK_SIZEX + tCol;

		// note can be in any row of blocks
		if (tRow < d_gates.size(0) && c < d_gates.size(1))
		{
			// only 1 in four threads does the point-wise part
			if (tCol % 4 == 0)
			{
				int output_index = tRow * state_size + (c/4);
				// vectorized load
				scalar_t4<scalar_t> gate_vals = reinterpret_cast<scalar_t4<scalar_t>*>(gate_weights.data())[output_index];
				// load other values too
				scalar_t grad_h_val = grad_h.data()[output_index];
				scalar_t grad_cell_val = grad_cell.data()[output_index];
				scalar_t new_cell_val = new_cell.data()[output_index];
				scalar_t old_cell_val = old_cell.data()[output_index];
				scalar_t forget_gate_val = forget_gate.data()[output_index];

				// calculate d_new_cell value (doesnt need to be stored)
				scalar_t d_new_cell = (grad_h_val * d_sigmoid(new_cell_val)) + grad_cell_val;

				// d_input_gate pre-activation
				gate_vals.input = -d_new_cell * d_sigmoid(gate_vals.input);
				// d_output_gate pre-activation
				gate_vals.output = -grad_h_val * d_sigmoid(gate_vals.output);
				// d_forget_gate pre-activation
				gate_vals.forget = (d_new_cell * old_cell_val) * d_sigmoid(gate_vals.forget);
				// d_candidate_cell pre-activation
				gate_vals.candidate = d_new_cell * d_sigmoid(gate_vals.candidate);

				// vectorized store outputs in shared memory for later
				reinterpret_cast<scalar_t4<scalar_t>*>(Bs)[(tRow*BLOCK_SIZEX + tCol) / 4] = gate_vals;

				// only the first row of blocks needs to store outputs to global
				if (by == 0) {
					d_old_cell.data()[output_index] = d_new_cell * forget_gate_val;
					// vectorized store to d_gates
					reinterpret_cast<scalar_t4<scalar_t>*>(d_gates.data())[output_index] = gate_vals;

					// defer d_bias calculation until we loop through Bs in second part of algorithm
				}
			}
		} else {
				// fill other elements with zeros
				Bs[tRow*BLOCK_SIZEX + tCol] = scalar_t();
		}

		// Make sure everything is done
		__syncthreads();

/**
		scalar_t d_bias_val = 0;

		#pragma unroll
		    for (int k = 0; k < BLOCK_SIZEY; ++k)
		    {
						// Do d-bias summation. Since whole columns of d_gates is captured in a
						// single block, we can do the bias calculation in one go
						// defer storing the output for later
						d_bias_val += Bs[k*BLOCK_SIZEX + tCol];
		    }

				int n = by * BLOCK_SIZEY + tRow;

				if (n == 0 && c < C.size(1)) {
						d_bias[c] = d_bias_val;
				}
**/

  // from mmAdd but modified a fair bit:
  // * - Bs is now fixed and filled by the above code
  // * - As is larger - (BLOCK_SIZEY, BLOCK_SIZEY)
	// * - first loop As is filled BLOCK_SIZEX x BLOCK_SIZEY (THREADS_PER_BLOCK) elements at a time
  // * - second loop every thread does its dot product and puts the result into global
	// * - loop includes d_bias calculation

	// Declaration of the shared memory array
	// NOTICE it's X,Y which means As is rotated in comparison to Bs
	__shared__ scalar_t As[BLOCK_SIZEX][BLOCK_SIZEY];

		// Index of the first element of the first sub-matrix of A
		// processed by the block
		int aBeginRow = by * BLOCK_SIZEX;

		int n = aBeginRow + tRow;

		// Load all of A in blocks of THREADS_PER_BLOCK
		// Notice we iterate by BLOCK_SIZEX rows because the tile is rotated for As
	#pragma unroll
		for (int firstRowInBatch = 0; firstRowInBatch < BLOCK_SIZEY; firstRowInBatch += BLOCK_SIZEX)
		{
			// Load the matrices from global to shared memory; each thread loads one element
			// NOTE: cols/rows that threads correspond to (in C) are flipped for this load
			//       but As is still filled row major
			int aRow = aBeginRow + firstRowInBatch + tCol;
			// int col = tRow;
			As[tCol][tRow] = (aRow < A.size(0) && tRow < A.size(1)) ?
																					A[aRow][tRow] : scalar_t(); // 0

			// Synchronize to make sure the matrices are loaded
			__syncthreads();

			// Accumulated output for the C value at
			// C[by * BLOCK_SIZEY + tRow][bx * BLOCK_SIZEX + tCol]
			scalar_t Csub = 0;
			scalar_t d_bias_val = 0;

			if (tRow >= firstRowInBatch && tRow < firstRowInBatch + BLOCK_SIZEX) {
				// Multiply the two sub-matrices together
				#pragma unroll
				for (int k = 0; k < BLOCK_SIZEY; ++k)
				{
						scalar_t Bs_val = Bs[k*BLOCK_SIZEX + tCol];
						Csub += As[tRow][k] * Bs_val;

						// Do d-bias summation. Since whole columns of d_gates is captured in a
						// single block, we can do the bias calculation in one go
						// defer storing the output for later
						d_bias_val += Bs_val;
				}

				// only store to global if within the boundaries of the output
				if (n < C.size(0) && c < C.size(1)) {
						C[n][c] = Csub;

						// d_bias only stored by 1 row of threads
						if (n == 0) {
							d_bias[c] = d_bias_val;
						}
				}
			}

		}

	}

/**
	// Declaration of the shared memory array
	__shared__ scalar_t As[BLOCK_SIZEY][BLOCK_SIZEY];

	  // Index of the first element of the first sub-matrix of A
		// processed by the block
	  int aBeginRow = by * BLOCK_SIZEY;

		int n = aBeginRow + tRow;


		// Load all of A in blocks of THREADS_PER_BLOCK
		// Notice we iterate by BLOCK_SIZEX rows because the tile is rotated for As
#pragma unroll
 	  for (int firstRowInBatch = 0; firstRowInBatch < BLOCK_SIZEY; firstRowInBatch += BLOCK_SIZEX)
		{
      // Load the matrices from global to shared memory; each thread loads one element
			// NOTE: cols/rows that threads correspond to (in C) are flipped for this load
			//       but As is still filled row major
			int asRow = firstRowInBatch + tCol;
			int aRow = aBeginRow + asRow;
			// int col = tRow;
			As[asRow][tRow] = (aRow < A.size(0) && tRow < A.size(1)) ?
																					A[aRow][tRow] : scalar_t(); // 0
		}

    // Synchronize to make sure the matrices are loaded
    __syncthreads();

	  // Accumulated output for the C value at
		// C[by * BLOCK_SIZEY + tRow][bx * BLOCK_SIZEX + tCol]
	  scalar_t Csub = 0;
		scalar_t d_bias_val = 0;

    // Multiply the two sub-matrices together
#pragma unroll
    for (int k = 0; k < BLOCK_SIZEY; ++k)
    {
				scalar_t Bs_val = Bs[k*BLOCK_SIZEX + tCol];
        Csub += As[tRow][k] * Bs_val;

				// Do d-bias summation. Since whole columns of d_gates is captured in a
				// single block, we can do the bias calculation in one go
				// defer storing the output for later
				d_bias_val += Bs_val;
    }

		// only store to global if within the boundaries of the output
		if (n < C.size(0) && c < C.size(1)) {
				C[n][c] = Csub;

				// d_bias only stored by 1 row of threads
				if (n == 0) {
					d_bias[c] = d_bias_val;
				}
		}

	}
**/
}

std::vector<torch::Tensor> forward_cuda
(
	torch::Tensor input,
	torch::Tensor old_h,
	torch::Tensor old_cell,
	torch::Tensor weightsT,
	torch::Tensor bias,
	int64_t threads,
	int64_t ONE_TO_MM
)
{
	at::cuda::CUDAStream cudaStream = at::cuda::getCurrentCUDAStream();
	cudaStream_t stream = (cudaStream_t) cudaStream;
	at::cuda::CUDAStreamGuard guard(cudaStream);

	auto options = weightsT.options().requires_grad(false);

  const auto batch_size = old_cell.size(0);
  const auto state_size = old_cell.size(1);

	auto X = torch::cat({old_h, input}, 1);

	//auto gate_weights = torch::addmm(bias, X, weightsT);
	auto gate_weights = torch::empty({X.size(0), weightsT.size(1)}, options);

/**
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
**/

	auto new_h = torch::empty({batch_size, state_size}, options);
  auto new_cell = torch::empty({batch_size, state_size}, options);
	auto forget_gate = torch::empty({batch_size, state_size}, options);

/**
	// Forward Point-wise with AoS memory layout
	//if (ONE_TO_MM == 0) {
		const dim3 blocks((state_size*batch_size + threads - 1) / threads, 1);

		AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "forward_cuda", ([&] {
			new_forward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
				batch_size*state_size,
				gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				forget_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
			);
		}));
	//}
**/

// if (threads == 256) {
		int t = 16;

		dim3 THREADS(t, t);
	  dim3 GRID((gate_weights.size(1) + t - 1) / t,
							(gate_weights.size(0) + t - 1) / t);

		AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "fused_forward", ([&] {
			fused_forward_cuda_kernel<scalar_t, 16, 256><<<GRID, THREADS, 0, stream>>>(
				gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				weightsT.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
				batch_size,
				state_size,
				old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				forget_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
			);
		}));

/**	} else if (threads == 1024) {
		int t = 32;

		dim3 THREADS(t, t);
	  dim3 GRID((gate_weights.size(1) + t - 1) / t,
							(gate_weights.size(0) + t - 1) / t);

		AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "custom_mmadd", ([&] {
			fused_forward_cuda_kernel<scalar_t, 32, 1024><<<GRID, THREADS, 0, stream>>>(
				gate_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				X.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				weightsT.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
				batch_size,
				state_size,
				old_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_h.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				new_cell.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
				forget_gate.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>()
			);
		}));
	}
**/

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
		torch::Tensor old_cell,
		int64_t threads,
		int64_t ONE_TO_MM
	)
	{
	at::cuda::CUDAStream cudaStream = at::cuda::getCurrentCUDAStream();
	cudaStream_t stream = (cudaStream_t) cudaStream;
	at::cuda::CUDAStreamGuard guard(cudaStream);

	// Store no gradients
	torch::NoGradGuard no_grad;
	auto options = grad_h.options().requires_grad(false);

  const auto batch_size = grad_h.size(0);
  const auto state_size = grad_h.size(1);

	// no need for .contiguous
	auto X_t = X.t();

	auto d_old_cell = torch::empty({old_cell.size(0), old_cell.size(1)}, options);
	auto d_gates = torch::empty({batch_size, 4*state_size}, options);
	auto d_bias = torch::zeros({4*state_size}, options);

	const dim3 blocks((state_size*batch_size + threads - 1) / threads, 1);

	AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
		new_backward_cuda_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
			batch_size,
			state_size,
			batch_size*state_size,
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

/**
	// only need empty now for d_bias since it's values are set not incremented
	auto d_bias = torch::empty({4*state_size}, options);
	torch::Tensor d_weights = torch::empty({X_t.size(0), d_gates.size(1)}, options);

	dim3 THREADS(20, 20);
	dim3 GRID((d_weights.size(1) + 20 - 1) / 20, (d_weights.size(0) + 20 - 1) / 20);
	//dim3 GRID((d_weights.size(1) + 4 - 1) / 4, 1);

	AT_DISPATCH_FLOATING_TYPES(grad_h.scalar_type(), "backward_cuda", ([&] {
		fused_backward_cuda_kernel<scalar_t, 20, 20, 400><<<GRID, THREADS,  0, stream>>>(
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
			d_bias.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
			X_t.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
			d_weights.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>()
		);
	}));

	//d_weights = X_t.mm(d_gates);
**/

	auto d_weights = X_t.mm(d_gates);
	auto d_X = d_gates.mm(weights.t());

	torch::Tensor d_old_h = d_X.slice(1, 0, state_size); // first state_size columns
	torch::Tensor d_input = d_X.slice(1, state_size); // from column [state_size + 1] to the end

	return {d_old_h, d_input, d_weights, d_bias, d_old_cell};
}
