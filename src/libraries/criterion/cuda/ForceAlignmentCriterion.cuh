/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda_runtime.h>

#include "libraries/criterion/Defines.h"

namespace w2l {
namespace cuda {

/// The numerator of ASG loss. Reference: https://arxiv.org/abs/1609.03193
template <class Float>
struct ForceAlignmentCriterion {
  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * L: target size
   */
  static size_t getWorkspaceSize(int B, int T, int N, int L);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * L: target size
   * scaleMode: type of size scaling
   * input: [B][T][N] input frames from network
   * target: [B][L] target labels
   * targetSize: [B] target sizes
   * trans: [N][N] transition matrix
   * loss: [B] (out) loss value
   * workspace: (in/out) internal workspace
   * stream: CUDA stream
   */
  static void forward(
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
      cudaStream_t stream);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * L: target size
   * target: [B][L] target labels
   * targetSize: [B] target sizes
   * grad: [B] gradient w.r.t. loss
   * inputGrad: [B][T][N] (out) gradient w.r.t. input
   * transGrad: [N][N] (out) gradient w.r.t. transitions
   * workspace: (in/out) internal workspace from forward
   * stream: CUDA stream
   */
  static void backward(
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
      cudaStream_t stream);

  static void viterbiPath(
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
      cudaStream_t stream);
};

} // namespace cuda
} // namespace w2l
