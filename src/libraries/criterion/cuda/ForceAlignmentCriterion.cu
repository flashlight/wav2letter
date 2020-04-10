/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cuda/ForceAlignmentCriterion.cuh"

#include <algorithm>
#include <cmath>

#include "libraries/common/CudaUtils.cuh"
#include "libraries/common/Workspace.h"
#include "libraries/criterion/cuda/CriterionUtils.cuh"

namespace {

template <class Float>
struct WorkspacePtrs {
  explicit WorkspacePtrs(void* workspace, int B, int T, int N, int L) {
    w2l::Workspace<> ws(workspace);
    ws.request(&scale, B);
    ws.request(&alpha, B, T, L);
    ws.request(&alphaGrad, B, T, L);
    ws.request(&transBatchGrad, B, N, N);
    ws.request(&transBuf1, B, L);
    ws.request(&transBuf2, B, L);
    ws.request(&transBufGrad1, B, L);
    ws.request(&transBufGrad2, B, L);
    requiredSize = ws.requiredSize();
  }

  Float* scale;
  double* alpha;
  double* alphaGrad;
  Float* transBatchGrad;
  Float* transBuf1;
  Float* transBuf2;
  Float* transBufGrad1;
  Float* transBufGrad2;
  size_t requiredSize;
};

/*
 * B thread blocks
 * L threads/block (ideally)
 */
template <class Float>
__global__ void forwardKernel(
    int T,
    int N,
    int _L,
    const Float* _input,
    const int* _target,
    const int* targetSize,
    const Float* trans,
    Float* _loss,
    WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;
  auto* alpha = &ws.alpha[b * T * _L];
  auto* input = &_input[b * T * N];
  auto* target = &_target[b * _L];
  auto* transBuf1 = &ws.transBuf1[b * _L];
  auto* transBuf2 = &ws.transBuf2[b * _L];
  int L = targetSize[b];

  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    alpha[i] = i == 0 ? input[target[0]] : 0;
    transBuf1[i] = trans[target[i] * N + target[i]];
    transBuf2[i] = i > 0 ? trans[target[i] * N + target[i - 1]] : 0;
  }

  for (int t = 1; t < T; ++t) {
    auto* inputCur = &input[t * N];
    auto* alphaPrev = &alpha[(t - 1) * L];
    auto* alphaCur = &alpha[t * L];

    int high = t < L ? t : L;
    int low = T - t < L ? L - (T - t) : 1;

    __syncthreads();

    if (threadIdx.x == 0) {
      if (T - t >= L) {
        alphaCur[0] = alphaPrev[0] + transBuf1[0] + inputCur[target[0]];
      }
    } else if (threadIdx.x == 1) {
      if (t < L) {
        alphaCur[high] =
            alphaPrev[high - 1] + transBuf2[high] + inputCur[target[high]];
      }
    }

    for (int i = low + threadIdx.x; i < high; i += blockDim.x) {
      double s1 = alphaPrev[i] + transBuf1[i];
      double s2 = alphaPrev[i - 1] + transBuf2[i];
      // lse = logSumExp(s1, s2)
      double lse =
          s1 < s2 ? s2 + log(1 + exp(s1 - s2)) : s1 + log(1 + exp(s2 - s1));
      alphaCur[i] = lse + inputCur[target[i]];
    }
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    _loss[b] = alpha[T * L - 1] * ws.scale[b];
  }
}

/*
 * B thread blocks
 * L threads/block (ideally)
 */
template <class Float>
__global__ void backwardKernel(
    int T,
    int N,
    int _L,
    const int* _target,
    const int* targetSize,
    const Float* grad,
    Float* _inputGrad,
    Float* transGrad,
    WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;
  auto* alpha = &ws.alpha[b * T * _L];
  auto* alphaGrad = &ws.alphaGrad[b * T * _L];
  auto* inputGrad = &_inputGrad[b * T * N];
  auto* target = &_target[b * _L];
  auto* transBatchGrad = &ws.transBatchGrad[b * N * N];
  auto* transBuf1 = &ws.transBuf1[b * _L];
  auto* transBuf2 = &ws.transBuf2[b * _L];
  auto* transBufGrad1 = &ws.transBufGrad1[b * _L];
  auto* transBufGrad2 = &ws.transBufGrad2[b * _L];
  int L = targetSize[b];

  if (threadIdx.x == 0) {
    alphaGrad[T * L - 1] = 1;
  }

  for (int t = T - 1; t > 0; --t) {
    auto* inputCurGrad = &inputGrad[t * N];
    auto* alphaPrev = &alpha[(t - 1) * L];
    auto* alphaCurGrad = &alphaGrad[t * L];
    auto* alphaPrevGrad = &alphaGrad[(t - 1) * L];

    int high = t < L ? t : L;
    int low = T - t < L ? L - (T - t) : 1;

    int high1 = t < L ? t + 1 : L;
    int low1 = T - t < L ? L - (T - t) : 0;

    __syncthreads();

    for (int i = low1 + threadIdx.x; i < high1; i += blockDim.x) {
      atomicAdd(&inputCurGrad[target[i]], alphaCurGrad[i]);
    }

    if (threadIdx.x == 0) {
      if (T - t >= L) {
        atomicAdd(&alphaPrevGrad[0], alphaCurGrad[0]);
        transBufGrad1[0] += alphaCurGrad[0];
      }
    } else if (threadIdx.x == 1) {
      if (t < L) {
        atomicAdd(&alphaPrevGrad[high - 1], alphaCurGrad[high]);
        transBufGrad2[high] += alphaCurGrad[high];
      }
    }

    for (int i = low + threadIdx.x; i < high; i += blockDim.x) {
      double s1 = alphaPrev[i] + transBuf1[i];
      double s2 = alphaPrev[i - 1] + transBuf2[i];
      // d1, d2 = dLogSumExp(s1, s2)
      double d1, d2;
      if (s1 < s2) {
        d2 = 1 / (1 + exp(s1 - s2));
        d1 = 1 - d2;
      } else {
        d1 = 1 / (1 + exp(s2 - s1));
        d2 = 1 - d1;
      }
      atomicAdd(&alphaPrevGrad[i], d1 * alphaCurGrad[i]);
      atomicAdd(&alphaPrevGrad[i - 1], d2 * alphaCurGrad[i]);
      transBufGrad1[i] += d1 * alphaCurGrad[i];
      transBufGrad2[i] += d2 * alphaCurGrad[i];
    }
  }

  __syncthreads();

  __shared__ Float gradScale;

  if (threadIdx.x == 0) {
    inputGrad[target[0]] += alphaGrad[0];
    gradScale = grad[b] * ws.scale[b];
  }

  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    atomicAdd(&transBatchGrad[target[i] * N + target[i]], transBufGrad1[i]);
    if (i > 0) {
      atomicAdd(
          &transBatchGrad[target[i] * N + target[i - 1]], transBufGrad2[i]);
    }
  }

  __syncthreads();

  for (int i = threadIdx.x; i < T * N; i += blockDim.x) {
    inputGrad[i] *= gradScale;
  }

  for (int i = threadIdx.x; i < N * N; i += blockDim.x) {
    atomicAdd(&transGrad[i], gradScale * transBatchGrad[i]);
  }
}

template <class Float>
__global__ void viterbiPathKernel(
    int T,
    int N,
    int _L,
    const Float* _input,
    const int* _target,
    const int* targetSize,
    const Float* trans,
    int* bestPaths,
    WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;
  auto* alpha = &ws.alpha[b * T * _L];
  auto* input = &_input[b * T * N];
  auto* target = &_target[b * _L];
  auto* transBuf1 = &ws.transBuf1[b * _L];
  auto* transBuf2 = &ws.transBuf2[b * _L];
  int L = targetSize[b];

  for (int i = threadIdx.x; i < L * T; i += blockDim.x) {
    alpha[i] = i == 0 ? input[target[0]] : -CUDART_INF_F;
  }

  for (int i = threadIdx.x; i < L; i += blockDim.x) {
    transBuf1[i] = trans[target[i] * N + target[i]];
    transBuf2[i] = i > 0 ? trans[target[i] * N + target[i - 1]] : 0;
  }
  if (L > T || L == 0) {
    return;
  }

  for (int t = 1; t < T; ++t) {
    auto* inputCur = &input[t * N];
    auto* alphaPrev = &alpha[(t - 1) * L];
    auto* alphaCur = &alpha[t * L];

    int high = t < L ? t : L;
    int low = T - t < L ? L - (T - t) : 1;

    // Ensure that all previous alphas have been computed
    __syncthreads();

    if (threadIdx.x == 0) {
      if (T - t >= L) {
        alphaCur[0] = alphaPrev[0] + transBuf1[0] + inputCur[target[0]];
      }
    } else if (threadIdx.x == 1) {
      if (t < L) {
        alphaCur[high] =
            alphaPrev[high - 1] + transBuf2[high] + inputCur[target[high]];
      }
    }

    for (int i = low + threadIdx.x; i < high; i += blockDim.x) {
      double s1 = alphaPrev[i] + transBuf1[i];
      double s2 = alphaPrev[i - 1] + transBuf2[i];
      alphaCur[i] = inputCur[target[i]] + max(s1, s2);
    }
  }
  // Ensure all threads are finished and alphas have been computed before
  // computing backward path
  __syncthreads();
  if (threadIdx.x == 0) {
    int ltrIdx = L - 1;
    for (int t = T - 1; t > 0; t--) {
      bestPaths[t + (b * T)] = target[ltrIdx];
      auto* alphaPrev = &alpha[(t - 1) * L];
      if (ltrIdx > 0) {
        double s1 = alphaPrev[ltrIdx] + transBuf1[ltrIdx];
        double s2 = alphaPrev[ltrIdx - 1] + transBuf2[ltrIdx];
        if (s2 > s1) {
          ltrIdx--;
        }
      }
    }
    bestPaths[b * T] = target[ltrIdx];
  }
}

} // namespace

namespace w2l {
namespace cuda {

template <class Float>
size_t
ForceAlignmentCriterion<Float>::getWorkspaceSize(int B, int T, int N, int L) {
  return WorkspacePtrs<Float>(nullptr, B, T, N, L).requiredSize;
}

template <class Float>
void ForceAlignmentCriterion<Float>::forward(
    int B,
    int T,
    int N,
    int L,
    CriterionScaleMode scaleMode,
    const Float* input,
    const int* target,
    const int* targetSize,
    const Float* trans,
    Float* loss,
    void* workspace,
    cudaStream_t stream) {
  int blockSize = std::min(256, (L + 31) / 32 * 32);
  WorkspacePtrs<Float> ws(workspace, B, T, N, L);
  CriterionUtils<Float>::computeScale(
      B, T, N, scaleMode, targetSize, ws.scale, stream);
  forwardKernel<<<B, blockSize, 0, stream>>>(
      T, N, L, input, target, targetSize, trans, loss, ws);
}

template <class Float>
void ForceAlignmentCriterion<Float>::backward(
    int B,
    int T,
    int N,
    int L,
    const int* target,
    const int* targetSize,
    const Float* grad,
    Float* inputGrad,
    Float* transGrad,
    void* workspace,
    cudaStream_t stream) {
  int blockSize = std::min(256, (L + 31) / 32 * 32);
  WorkspacePtrs<Float> ws(workspace, B, T, N, L);
  setZero(inputGrad, B * T * N, stream);
  setZero(transGrad, N * N, stream);
  setZero(ws.alphaGrad, B * T * L, stream);
  setZero(ws.transBatchGrad, B * N * N, stream);
  setZero(ws.transBufGrad1, B * L, stream);
  setZero(ws.transBufGrad2, B * L, stream);
  backwardKernel<<<B, blockSize, 0, stream>>>(
      T, N, L, target, targetSize, grad, inputGrad, transGrad, ws);
}

template <class Float>
void ForceAlignmentCriterion<Float>::viterbiPath(
    int B,
    int T,
    int N,
    int L,
    const Float* input,
    const int* target,
    const int* targetSize,
    const Float* trans,
    int* bestPaths,
    void* workspace,
    cudaStream_t stream) {
  int blockSize = std::min(256, (L + 31) / 32 * 32);
  WorkspacePtrs<Float> ws(workspace, B, T, N, L);
  setZero(ws.alpha, B * T * L, stream);
  viterbiPathKernel<<<B, blockSize, 0, stream>>>(
      T, N, L, input, target, targetSize, trans, bestPaths, ws);
}

template struct ForceAlignmentCriterion<float>;
template struct ForceAlignmentCriterion<double>;

} // namespace cuda
} // namespace w2l
