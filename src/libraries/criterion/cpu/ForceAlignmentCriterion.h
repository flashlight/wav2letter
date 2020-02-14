/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include "libraries/criterion/Defines.h"

namespace w2l {
namespace cpu {

/// Check CUDA header for docs.
template <class Float>
struct ForceAlignmentCriterion {
  static size_t getWorkspaceSize(int B, int T, int N, int L);

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
      void* workspace);

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
      void* workspace);

  static void viterbi(
      int B,
      int T,
      int N,
      int L,
      const Float* input,
      const int* target,
      const int* targetSize,
      const Float* trans,
      int* bestPaths,
      void* workspace);
};

} // namespace cpu
} // namespace w2l
