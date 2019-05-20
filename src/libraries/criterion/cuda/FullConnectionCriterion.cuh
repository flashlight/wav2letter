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

/// The denominator of ASG loss. Reference: https://arxiv.org/abs/1609.03193
template <class Float>
struct FullConnectionCriterion {
  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   */
  static size_t getWorkspaceSize(int B, int T, int N);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * scaleMode: type of size scaling
   * input: [B][T][N] input frames from network
   * targetSize: [B] target sizes (may be null if not needed for scaleMode)
   * trans: [N][N] transition matrix
   * loss: [B] (out) loss value
   * workspace: (in/out) internal workspace
   * stream: CUDA stream
   */
  static void forward(
      int B,
      int T,
      int N,
      CriterionScaleMode scaleMode,
      const Float* input,
      const int* targetSize,
      const Float* trans,
      Float* loss,
      void* workspace,
      cudaStream_t stream);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * trans: [N][N] transition matrix
   * grad: [B] gradient w.r.t. loss
   * inputGrad: [B][T][N] (out) gradient w.r.t. input
   * transGrad: [N][N] (out) gradient w.r.t transitions
   * workspace: (in/out) internal workspace from forward
   * stream: CUDA stream
   */
  static void backward(
      int B,
      int T,
      int N,
      const Float* trans,
      const Float* grad,
      Float* inputGrad,
      Float* transGrad,
      void* workspace,
      cudaStream_t stream);
};

} // namespace cuda
} // namespace w2l
