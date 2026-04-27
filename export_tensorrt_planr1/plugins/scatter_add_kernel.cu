/*
 * ScatterAdd CUDA Kernel for TensorRT Plugin
 * 
 * Implements: output[index[i]] += src[i] for all i
 * This is equivalent to PyTorch's scatter_add_ operation.
 */

#include "scatter_add_kernel.h"
#include <cuda_fp16.h>

namespace planr1 {
namespace plugin {

// ============================================================================
// FP32 Kernel
// ============================================================================

__global__ void scatter_add_fp32_kernel(
    const float* __restrict__ src,      // [E, D] source values
    const int64_t* __restrict__ index,  // [E] target indices
    float* __restrict__ output,         // [N, D] output (accumulated)
    int E,                              // number of edges
    int D,                              // feature dimension
    int N                               // number of nodes
) {
    // Each thread handles one (edge, feature) pair
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        int64_t dst_idx = index[e];
        
        // Bounds check
        if (dst_idx >= 0 && dst_idx < N) {
            float value = src[e * D + d];
            atomicAdd(&output[dst_idx * D + d], value);
        }
    }
}

// ============================================================================
// FP16 Kernel (for Tensor Core acceleration)
// ============================================================================

__global__ void scatter_add_fp16_kernel(
    const __half* __restrict__ src,     // [E, D] source values (FP16)
    const int64_t* __restrict__ index,  // [E] target indices
    __half* __restrict__ output,        // [N, D] output (FP16)
    int E,
    int D,
    int N
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        int64_t dst_idx = index[e];
        
        if (dst_idx >= 0 && dst_idx < N) {
            // FP16 atomicAdd requires compute capability >= 7.0
            #if __CUDA_ARCH__ >= 700
                atomicAdd(&output[dst_idx * D + d], src[e * D + d]);
            #else
                // Fallback: convert to float, add, convert back
                float val = __half2float(src[e * D + d]);
                float* out_ptr = reinterpret_cast<float*>(&output[dst_idx * D + d]);
                // Note: This is not truly atomic for FP16 on older architectures
                atomicAdd(out_ptr, val);
            #endif
        }
    }
}

// ============================================================================
// Launcher Functions
// ============================================================================

void launchScatterAddFP32(
    const float* src,
    const int64_t* index,
    float* output,
    int E, int D, int N,
    cudaStream_t stream
) {
    // Zero out output first (scatter_add accumulates into output)
    cudaMemsetAsync(output, 0, N * D * sizeof(float), stream);
    
    if (E == 0) return;  // No edges to process
    
    // Configure grid and block dimensions
    // Use 2D blocks for better occupancy
    dim3 block(32, 8);  // 256 threads per block
    dim3 grid(
        (E + block.x - 1) / block.x,
        (D + block.y - 1) / block.y
    );
    
    scatter_add_fp32_kernel<<<grid, block, 0, stream>>>(
        src, index, output, E, D, N
    );
}

void launchScatterAddFP32NoZero(
    const float* src,
    const int64_t* index,
    float* output,
    int E, int D, int N,
    cudaStream_t stream
) {
    // Don't zero output - caller has already set it up with base data
    if (E == 0) return;
    
    dim3 block(32, 8);
    dim3 grid(
        (E + block.x - 1) / block.x,
        (D + block.y - 1) / block.y
    );
    
    scatter_add_fp32_kernel<<<grid, block, 0, stream>>>(
        src, index, output, E, D, N
    );
}

// ============================================================================
// 2D Indices Kernels (for ONNX ScatterElements with expanded indices)
// ============================================================================

// ONNX ScatterElements with axis=0 has indices shape [E, D]
// where indices[e, d] = target row for src[e, d]
// For scatter_add on graph attention, all indices in row e are the same
__global__ void scatter_add_fp32_2d_kernel(
    const float* __restrict__ src,        // [E, D] values to scatter
    const int64_t* __restrict__ indices,  // [E, D] expanded indices
    float* __restrict__ output,           // [N, D] output (accumulated)
    int E, int D, int N
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        // Get the target index for this (e, d) position
        int64_t dst_idx = indices[e * D + d];
        
        if (dst_idx >= 0 && dst_idx < N) {
            float value = src[e * D + d];
            atomicAdd(&output[dst_idx * D + d], value);
        }
    }
}

void launchScatterAddFP32_2D(
    const float* src,
    const int64_t* indices,  // [E, D] expanded indices
    float* output,
    int E, int D, int N,
    cudaStream_t stream
) {
    // Don't zero output - it already contains base data
    if (E == 0) return;
    
    dim3 block(32, 8);
    dim3 grid(
        (E + block.x - 1) / block.x,
        (D + block.y - 1) / block.y
    );
    
    scatter_add_fp32_2d_kernel<<<grid, block, 0, stream>>>(
        src, indices, output, E, D, N
    );
}

// FP16 version with 2D indices
__global__ void scatter_add_fp16_2d_kernel(
    const __half* __restrict__ src,
    const int64_t* __restrict__ indices,
    float* __restrict__ output,  // FP32 accumulator
    int E, int D, int N
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        int64_t dst_idx = indices[e * D + d];
        
        if (dst_idx >= 0 && dst_idx < N) {
            float value = __half2float(src[e * D + d]);
            atomicAdd(&output[dst_idx * D + d], value);
        }
    }
}

__global__ void convert_and_add_base_kernel(
    const __half* __restrict__ base,  // FP16 base data
    float* __restrict__ output,       // FP32 output
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __half2float(base[idx]);
    }
}

void launchScatterAddFP16_2D(
    const __half* src,
    const int64_t* indices,
    __half* output,
    float* workspace,  // [N, D] FP32 workspace
    int E, int D, int N,
    cudaStream_t stream
) {
    // Step 1: Convert base data (output) to FP32 workspace
    int total = N * D;
    int block1d = 256;
    int grid1d = (total + block1d - 1) / block1d;
    convert_and_add_base_kernel<<<grid1d, block1d, 0, stream>>>(
        output, workspace, total
    );
    
    // Step 2: Accumulate in FP32
    if (E > 0) {
        dim3 block(32, 8);
        dim3 grid(
            (E + block.x - 1) / block.x,
            (D + block.y - 1) / block.y
        );
        scatter_add_fp16_2d_kernel<<<grid, block, 0, stream>>>(
            src, indices, workspace, E, D, N
        );
    }
    
    // Step 3: Convert back to FP16
    extern __global__ void convert_fp32_to_fp16_kernel(const float*, __half*, int);
    convert_fp32_to_fp16_kernel<<<grid1d, block1d, 0, stream>>>(
        workspace, output, total
    );
}

void launchScatterAddFP16(
    const __half* src,
    const int64_t* index,
    __half* output,
    int E, int D, int N,
    cudaStream_t stream
) {
    // Zero out output
    cudaMemsetAsync(output, 0, N * D * sizeof(__half), stream);
    
    if (E == 0) return;
    
    dim3 block(32, 8);
    dim3 grid(
        (E + block.x - 1) / block.x,
        (D + block.y - 1) / block.y
    );
    
    scatter_add_fp16_kernel<<<grid, block, 0, stream>>>(
        src, index, output, E, D, N
    );
}

// ============================================================================
// Mixed Precision: FP16 input, FP32 accumulation, FP16 output
// (More numerically stable for large aggregations)
// ============================================================================

__global__ void scatter_add_mixed_kernel(
    const __half* __restrict__ src,     // [E, D] FP16 source
    const int64_t* __restrict__ index,  // [E] indices
    float* __restrict__ output,         // [N, D] FP32 accumulator
    int E, int D, int N
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (e < E && d < D) {
        int64_t dst_idx = index[e];
        
        if (dst_idx >= 0 && dst_idx < N) {
            float value = __half2float(src[e * D + d]);
            atomicAdd(&output[dst_idx * D + d], value);
        }
    }
}

__global__ void convert_fp32_to_fp16_kernel(
    const float* __restrict__ input,
    __half* __restrict__ output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = __float2half(input[idx]);
    }
}

void launchScatterAddMixed(
    const __half* src,
    const int64_t* index,
    __half* output,
    float* workspace,  // [N, D] FP32 workspace
    int E, int D, int N,
    cudaStream_t stream
) {
    // Zero workspace
    cudaMemsetAsync(workspace, 0, N * D * sizeof(float), stream);
    
    if (E > 0) {
        dim3 block(32, 8);
        dim3 grid(
            (E + block.x - 1) / block.x,
            (D + block.y - 1) / block.y
        );
        
        scatter_add_mixed_kernel<<<grid, block, 0, stream>>>(
            src, index, workspace, E, D, N
        );
    }
    
    // Convert back to FP16
    int total = N * D;
    int block1d = 256;
    int grid1d = (total + block1d - 1) / block1d;
    convert_fp32_to_fp16_kernel<<<grid1d, block1d, 0, stream>>>(
        workspace, output, total
    );
}

}  // namespace plugin
}  // namespace planr1
