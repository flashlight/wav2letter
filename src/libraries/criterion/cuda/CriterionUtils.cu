/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cuda/CriterionUtils.cuh"

#include <algorithm>

namespace {

/*
 * B thread blocks
 * 32 threads/block (ideally)
 */
__global__ void
batchTargetSizeKernel(int L, int maxSize, const int* _target, int* targetSize) {
  int b = blockIdx.x;
  auto target = _target + b * L;

  __shared__ int idx;

  if (threadIdx.x == 0) {
    idx = 0;
  }

  __syncthreads();

  for (int i = L - 1 - threadIdx.x; i >= 0; i -= blockDim.x) {
    if (target[i] >= 0) {
      atomicMax(&idx, i + 1);
      break;
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    targetSize[b] = idx < maxSize ? idx : maxSize;
  }
}

/*
 * 1 thread block
 * B threads/block (ideally)
 */
template <class Float>
__global__ void computeScaleKernel(
    int B,
    int T,
    int /* N */,
    w2l::CriterionScaleMode scaleMode,
    const int* targetSize,
    Float* scale) {
  for (int b = threadIdx.x; b < B; b += blockDim.x) {
    switch (scaleMode) {
      case w2l::CriterionScaleMode::NONE:
        scale[b] = 1.0;
        break;
      case w2l::CriterionScaleMode::INPUT_SZ:
        scale[b] = T > 0 ? 1.0 / T : 1.0;
        break;
      case w2l::CriterionScaleMode::INPUT_SZ_SQRT:
        scale[b] = T > 0 ? std::sqrt(1.0 / T) : 1.0;
        break;
      case w2l::CriterionScaleMode::TARGET_SZ:
        scale[b] = targetSize[b] > 0 ? 1.0 / targetSize[b] : 1.0;
        break;
      case w2l::CriterionScaleMode::TARGET_SZ_SQRT:
        scale[b] = targetSize[b] > 0 ? std::sqrt(1.0 / targetSize[b]) : 1.0;
        break;
      default:
        break;
    }
  }
}

} // namespace

namespace w2l {
namespace cuda {

template <class Float>
void CriterionUtils<Float>::batchTargetSize(
    int B,
    int L,
    int maxSize,
    const int* target,
    int* targetSize,
    cudaStream_t stream) {
  batchTargetSizeKernel<<<B, 32, 0, stream>>>(L, maxSize, target, targetSize);
}

template <class Float>
void CriterionUtils<Float>::computeScale(
    int B,
    int T,
    int N,
    CriterionScaleMode scaleMode,
    const int* targetSize,
    Float* scale,
    cudaStream_t stream) {
  int blockSize = std::min(256, (B + 31) / 32 * 32);
  computeScaleKernel<<<1, blockSize, 0, stream>>>(
      B, T, N, scaleMode, targetSize, scale);
}

template struct CriterionUtils<float>;
template struct CriterionUtils<double>;

} // namespace cuda
} // namespace w2l
