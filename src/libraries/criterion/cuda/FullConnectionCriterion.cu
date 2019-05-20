/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cuda/FullConnectionCriterion.cuh"

#include <cmath>

#include <cub/cub.cuh>

#include "libraries/common/CudaUtils.cuh"
#include "libraries/common/Workspace.h"
#include "libraries/criterion/cuda/CriterionUtils.cuh"

namespace {

constexpr int kBlockSize = 32;

template <class Float>
struct WorkspacePtrs {
  explicit WorkspacePtrs(void* workspace, int B, int T, int N) {
    w2l::Workspace<> ws(workspace);
    ws.request(&scale, B);
    ws.request(&alpha, B, T, N);
    ws.request(&alphaGrad, B, T, N);
    ws.request(&transBatchGrad, B, N, N);
    ws.request(&transBuf, B, N, N);
    requiredSize = ws.requiredSize();
  }

  Float* scale;
  double* alpha;
  double* alphaGrad;
  double* transBatchGrad;
  double* transBuf;
  size_t requiredSize;
};

/*
 * B thread blocks
 * kBlockSize threads/block
 */
template <class Float>
__global__ void
forwardInitial(int T, int N, const Float* input, WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    int k = b * T * N + n;
    ws.alpha[k] = input[k];
  }
}

/*
 * B * N thread blocks (B if Final)
 * kBlockSize threads/block
 */
template <bool Final, class Float>
__global__ void forwardStep(
    int T,
    int N,
    int t,
    const Float* input,
    const Float* trans,
    Float* loss,
    WorkspacePtrs<Float> ws) {
  int b, m;
  if (Final) {
    b = blockIdx.x;
  } else {
    b = blockIdx.x / N;
    m = blockIdx.x % N;
  }

  const auto* alphaPrev = &ws.alpha[b * T * N + (t - 1) * N];
  const auto* inputCur = &input[b * T * N + t * N];
  auto* alphaCur = &ws.alpha[b * T * N + t * N];
  auto* transBuf = &ws.transBuf[blockIdx.x * N];

  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double maxValue;

  double threadMax = -INFINITY;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    double val = transBuf[n] = alphaPrev[n] + (Final ? 0 : trans[m * N + n]);
    threadMax = val > threadMax ? val : threadMax;
  }

  double maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }

  __syncthreads();

  double threadSum = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    threadSum += exp(transBuf[n] - maxValue);
  }

  double sumResult = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    if (Final) {
      loss[b] = ws.scale[b] * (log(sumResult) + maxValue);
    } else {
      alphaCur[m] = log(sumResult) + maxValue + inputCur[m];
    }
  }
}

/*
 * B * N thread blocks (B if Initial)
 * kBlockSize threads/block
 */
template <bool Initial, class Float>
__global__ void backwardStep1(
    int T,
    int N,
    int t,
    const Float* trans,
    WorkspacePtrs<Float> ws) {
  int b, m;
  if (Initial) {
    b = blockIdx.x;
  } else {
    b = blockIdx.x / N;
    m = blockIdx.x % N;
  }

  const auto* alphaPrev = &ws.alpha[b * T * N + (t - 1) * N];
  const auto* alphaCurGrad = &ws.alphaGrad[b * T * N + t * N];
  auto* alphaPrevGrad = &ws.alphaGrad[b * T * N + (t - 1) * N];
  auto* transBuf = &ws.transBuf[blockIdx.x * N];
  auto* transBatchGrad = &ws.transBatchGrad[blockIdx.x * N];

  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;
  __shared__ double maxValue;
  __shared__ double sumValue;

  double threadMax = -INFINITY;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    double val = transBuf[n] = alphaPrev[n] + (Initial ? 0 : trans[m * N + n]);
    threadMax = val > threadMax ? val : threadMax;
  }

  double maxResult = BlockReduce(tempStorage).Reduce(threadMax, cub::Max());
  if (threadIdx.x == 0) {
    maxValue = maxResult;
  }

  double threadSum = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    transBuf[n] = exp(transBuf[n] - maxValue);
    threadSum += transBuf[n];
  }

  double sumResult = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    sumValue = sumResult;
  }

  __syncthreads();

  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    if (Initial) {
      alphaPrevGrad[n] = transBuf[n] / sumValue;
    } else {
      transBuf[n] = transBuf[n] / sumValue * alphaCurGrad[m];
      transBatchGrad[n] += transBuf[n];
    }
  }
}

/*
 * B * N thread blocks
 * kBlockSize threads/block
 */
template <class Float>
__global__ void backwardStep2(int T, int N, int t, WorkspacePtrs<Float> ws) {
  int b = blockIdx.x / N;
  int m = blockIdx.x % N;

  auto* alphaPrevGrad = &ws.alphaGrad[b * T * N + (t - 1) * N];

  using BlockReduce = cub::BlockReduce<double, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  double threadSum = 0;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    threadSum += ws.transBuf[b * N * N + n * N + m];
  }

  double sumResult = BlockReduce(tempStorage).Sum(threadSum);
  if (threadIdx.x == 0) {
    alphaPrevGrad[m] = sumResult;
  }
}

/*
 * B thread blocks
 * 128 threads/block
 */
template <class Float>
__global__ void backwardFinal(
    int T,
    int N,
    const Float* _grad,
    Float* _inputGrad,
    Float* transGrad,
    WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;

  auto* alphaGrad = &ws.alphaGrad[b * T * N];
  auto* inputGrad = &_inputGrad[b * T * N];
  auto* transBatchGrad = &ws.transBatchGrad[b * N * N];

  __shared__ Float gradScale;

  if (threadIdx.x == 0) {
    gradScale = ws.scale[b] * _grad[b];
  }

  __syncthreads();

  for (int i = threadIdx.x; i < T * N; i += blockDim.x) {
    inputGrad[i] = gradScale * alphaGrad[i];
  }

  for (int i = threadIdx.x; i < N * N; i += blockDim.x) {
    atomicAdd(&transGrad[i], gradScale * transBatchGrad[i]);
  }
}

} // namespace

namespace w2l {
namespace cuda {

template <class Float>
size_t FullConnectionCriterion<Float>::getWorkspaceSize(int B, int T, int N) {
  return WorkspacePtrs<Float>(nullptr, B, T, N).requiredSize;
}

template <class Float>
void FullConnectionCriterion<Float>::forward(
    int B,
    int T,
    int N,
    CriterionScaleMode scaleMode,
    const Float* input,
    const int* targetSize,
    const Float* trans,
    Float* loss,
    void* workspace,
    cudaStream_t stream) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);
  CriterionUtils<Float>::computeScale(
      B, T, N, scaleMode, targetSize, ws.scale, stream);
  forwardInitial<<<B, kBlockSize, 0, stream>>>(T, N, input, ws);
  for (int t = 1; t < T; ++t) {
    forwardStep<false>
        <<<B * N, kBlockSize, 0, stream>>>(T, N, t, input, trans, loss, ws);
  }
  forwardStep<true>
      <<<B, kBlockSize, 0, stream>>>(T, N, T, input, trans, loss, ws);
}

template <class Float>
void FullConnectionCriterion<Float>::backward(
    int B,
    int T,
    int N,
    const Float* trans,
    const Float* grad,
    Float* inputGrad,
    Float* transGrad,
    void* workspace,
    cudaStream_t stream) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);
  setZero(inputGrad, B * T * N, stream);
  setZero(transGrad, N * N, stream);
  setZero(ws.transBatchGrad, B * N * N, stream);
  backwardStep1<true><<<B, kBlockSize, 0, stream>>>(T, N, T, trans, ws);
  for (int t = T - 1; t > 0; --t) {
    backwardStep1<false><<<B * N, kBlockSize, 0, stream>>>(T, N, t, trans, ws);
    backwardStep2<<<B * N, kBlockSize, 0, stream>>>(T, N, t, ws);
  }
  backwardFinal<<<B, 128, 0, stream>>>(T, N, grad, inputGrad, transGrad, ws);
}

template struct FullConnectionCriterion<float>;
template struct FullConnectionCriterion<double>;

} // namespace cuda
} // namespace w2l
