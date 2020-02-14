/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cstddef>

namespace w2l {
namespace cpu {

template <class Float>
struct ConnectionistTemporalClassificationCriterion {
  static size_t getWorkspaceSize(int B, int T, int N, int L);

  static void viterbi(
      int B,
      int T,
      int N,
      int L,
      const Float* input,
      const int* target,
      const int* targetSize,
      int* bestPaths,
      void* workspace);
};
} // namespace cpu
} // namespace w2l
