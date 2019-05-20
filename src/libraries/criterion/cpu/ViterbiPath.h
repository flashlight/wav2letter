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

/// Check CUDA header for docs.
template <class Float>
struct ViterbiPath {
  static size_t getWorkspaceSize(int B, int T, int N);

  static void compute(
      int B,
      int T,
      int N,
      const Float* input,
      const Float* trans,
      int* path,
      void* workspace);
};

} // namespace cpu
} // namespace w2l
