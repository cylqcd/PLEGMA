// CUDA kernels backing PLEGMA_QLoops_fast.
//
// The kernels operate on flat real arrays (length n = field_length *
// total_length * 2).  Float is the underlying real precision (float
// or double).

#include "PLEGMA_QLoops_fast.h"

#include <cuda_runtime.h>
#include <cstddef>

namespace bhu {

namespace {

constexpr int kBlockSize = 256;

template <typename Float>
__global__ void axpy_real_kernel(Float* __restrict__ acc,
                                 const Float* __restrict__ src,
                                 Float val,
                                 std::size_t n)
{
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        acc[i] += val * src[i];
    }
}

template <typename Float>
__global__ void scal_real_kernel(Float* __restrict__ dst,
                                 const Float* __restrict__ src,
                                 Float val,
                                 std::size_t n)
{
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = tid; i < n; i += stride) {
        dst[i] = val * src[i];
    }
}

inline int num_blocks(std::size_t n)
{
    // Saturate the device a couple of times; the kernel is grid-stride.
    constexpr int kMaxBlocks = 65535;
    std::size_t needed = (n + kBlockSize - 1) / kBlockSize;
    if (needed > static_cast<std::size_t>(kMaxBlocks)) return kMaxBlocks;
    return static_cast<int>(needed);
}

}  // namespace

template <typename Float>
void launch_axpy_real(Float* acc, const Float* src, Float val, std::size_t n)
{
    if (n == 0) return;
    axpy_real_kernel<Float><<<num_blocks(n), kBlockSize>>>(acc, src, val, n);
}

template <typename Float>
void launch_scal_real(Float* dst, const Float* src, Float val, std::size_t n)
{
    if (n == 0) return;
    scal_real_kernel<Float><<<num_blocks(n), kBlockSize>>>(dst, src, val, n);
}

template void launch_axpy_real<float >(float*,  const float*,  float,  std::size_t);
template void launch_axpy_real<double>(double*, const double*, double, std::size_t);
template void launch_scal_real<float >(float*,  const float*,  float,  std::size_t);
template void launch_scal_real<double>(double*, const double*, double, std::size_t);

}  // namespace bhu
