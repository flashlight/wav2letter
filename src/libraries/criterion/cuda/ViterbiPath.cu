/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cuda/ViterbiPath.cuh"

#include <cmath>

#include <cub/cub.cuh>

#include "libraries/common/Workspace.h"

namespace {

constexpr int kBlockSize = 32;

template <class Float>
struct WorkspacePtrs {
  explicit WorkspacePtrs(void* workspace, int B, int T, int N) {
    w2l::Workspace<> ws(workspace);
    ws.request(&alpha, B, 2, N);
    ws.request(&beta, B, T, N);
    requiredSize = ws.requiredSize();
  }

  Float* alpha;
  int* beta;
  size_t requiredSize;
};

/*
 * B thread blocks
 * kBlockSize threads/block
 */
template <class Float>
__global__ void
computeInitial(int T, int N, const Float* input, WorkspacePtrs<Float> ws) {
  int b = blockIdx.x;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    ws.alpha[b * 2 * N + n] = input[b * T * N + n];
  }
}

/*
 * B * N thread blocks (B if Final)
 * kBlockSize threads/block
 */
template <bool Final, class Float>
__global__ void computeStep(
    int T,
    int N,
    int t,
    const Float* input,
    const Float* trans,
    int* _path,
    WorkspacePtrs<Float> ws) {
  int b, m;
  if (Final) {
    b = blockIdx.x;
  } else {
    b = blockIdx.x / N;
    m = blockIdx.x % N;
  }

  const auto* alphaPrev = &ws.alpha[b * 2 * N + ((t - 1) % 2) * N];
  const auto* inputCur = &input[b * T * N + t * N];
  auto* alphaCur = &ws.alpha[b * 2 * N + (t % 2) * N];
  auto* betaCur = &ws.beta[b * T * N + t * N];

  using BlockReduce =
      cub::BlockReduce<cub::KeyValuePair<int, Float>, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  cub::KeyValuePair<int, Float> threadMax;
  threadMax.value = -INFINITY;
  for (int n = threadIdx.x; n < N; n += blockDim.x) {
    Float val = alphaPrev[n] + (Final ? 0 : trans[m * N + n]);
    if (val > threadMax.value) {
      threadMax.key = n;
      threadMax.value = val;
    }
  }

  auto result = BlockReduce(tempStorage).Reduce(threadMax, cub::ArgMax());
  if (threadIdx.x == 0) {
    if (Final) {
      auto* path = &_path[b * T];
      path[T - 1] = result.key;
      for (int s = T - 1; s > 0; --s) {
        path[s - 1] = ws.beta[b * T * N + s * N + path[s]];
      }
    } else {
      alphaCur[m] = result.value + inputCur[m];
      betaCur[m] = result.key;
    }
  }
}

} // namespace

namespace w2l {
namespace cuda {

template <class Float>
size_t ViterbiPath<Float>::getWorkspaceSize(int B, int T, int N) {
  return WorkspacePtrs<Float>(nullptr, B, T, N).requiredSize;
}

template <class Float>
void ViterbiPath<Float>::compute(
    int B,
    int T,
    int N,
    const Float* input,
    const Float* trans,
    int* path,
    void* workspace,
    cudaStream_t stream) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);
  computeInitial<<<B, kBlockSize, 0, stream>>>(T, N, input, ws);
  for (int t = 1; t < T; ++t) {
    computeStep<false>
        <<<B * N, kBlockSize, 0, stream>>>(T, N, t, input, trans, path, ws);
  }
  computeStep<true>
      <<<B, kBlockSize, 0, stream>>>(T, N, T, input, trans, path, ws);
}

template struct ViterbiPath<float>;
template struct ViterbiPath<double>;

} // namespace cuda
} // namespace w2l
