/*
 * ScatterAdd CUDA Kernel Header
 */

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace planr1 {
namespace plugin {

/**
 * Launch FP32 scatter_add kernel
 * 
 * Computes: output[index[e], :] += src[e, :] for all e in [0, E)
 * 
 * @param src     [E, D] source tensor
 * @param index   [E] target indices (values in [0, N))
 * @param output  [N, D] output tensor (will be zeroed first)
 * @param E       number of edges
 * @param D       feature dimension
 * @param N       number of nodes
 * @param stream  CUDA stream
 */
void launchScatterAddFP32(
    const float* src,
    const int64_t* index,
    float* output,
    int E, int D, int N,
    cudaStream_t stream
);

/**
 * Launch FP32 scatter_add kernel without zeroing output
 * Use when output already contains base data
 */
void launchScatterAddFP32NoZero(
    const float* src,
    const int64_t* index,
    float* output,
    int E, int D, int N,
    cudaStream_t stream
);

/**
 * Launch FP16 scatter_add kernel
 * Uses native FP16 atomicAdd (requires compute capability >= 7.0)
 */
void launchScatterAddFP16(
    const __half* src,
    const int64_t* index,
    __half* output,
    int E, int D, int N,
    cudaStream_t stream
);

/**
 * Launch mixed-precision scatter_add kernel
 * FP16 input → FP32 accumulation → FP16 output
 * More numerically stable for large aggregations
 * 
 * @param workspace  [N, D] FP32 workspace buffer
 */
void launchScatterAddMixed(
    const __half* src,
    const int64_t* index,
    __half* output,
    float* workspace,
    int E, int D, int N,
    cudaStream_t stream
);

/**
 * Calculate required workspace size for mixed precision mode
 */
inline size_t getScatterAddWorkspaceSize(int N, int D) {
    return N * D * sizeof(float);
}

/**
 * Launch FP32 scatter_add with 2D expanded indices (ONNX ScatterElements format)
 * indices shape: [E, D] where indices[e, d] is target row for src[e, d]
 */
void launchScatterAddFP32_2D(
    const float* src,
    const int64_t* indices,  // [E, D] expanded
    float* output,
    int E, int D, int N,
    cudaStream_t stream
);

/**
 * Launch FP16 scatter_add with 2D expanded indices
 * Uses FP32 accumulation for numerical stability
 */
void launchScatterAddFP16_2D(
    const __half* src,
    const int64_t* indices,
    __half* output,
    float* workspace,
    int E, int D, int N,
    cudaStream_t stream
);

}  // namespace plugin
}  // namespace planr1
