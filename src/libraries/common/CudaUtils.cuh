/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>

#include <cuda.h>
#include <math_constants.h> // for CUDART_INF

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600

/// Double-precision `atomicAdd` backport for compute capability < 6.0
/// From NVIDIA docs: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
static __inline__ __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
        address_as_ull,
        assumed,
        __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}

#endif

namespace w2l {
namespace cuda {

/// Zeroes `count * sizeof(T)` device bytes
template <typename T>
void setZero(T* devPtr, size_t count, cudaStream_t stream) {
  cudaMemsetAsync(devPtr, 0, count * sizeof(T), stream);
}

} // namespace cuda
} // namespace w2l
