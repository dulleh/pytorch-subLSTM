#pragma once

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z);

template <typename scalar_t>
__device__ __forceinline__ scalar_t d_sigmoid(scalar_t z);

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
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell);