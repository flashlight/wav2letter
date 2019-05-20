/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libraries/criterion/Defines.h"

namespace w2l {
namespace cpu {

/// Check CUDA header for docs.
template <class Float>
struct CriterionUtils {
  static void batchTargetSize(
      int B,
      int L,
      int maxSize,
      const int* target,
      int* targetSize);

  static void computeScale(
      int B,
      int T,
      int N,
      CriterionScaleMode scaleMode,
      const int* targetSize,
      Float* scale);
};

} // namespace cpu
} // namespace w2l
