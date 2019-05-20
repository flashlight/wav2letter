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

template <class Float>
struct CriterionUtils {
  /**
   * B: batch size
   * L: target size
   * maxSize: target size results are clamped down to this
   * target: [B][L] target labels
   * targetSize: [B] (out) target sizes
   * stream: CUDA stream
   */
  static void batchTargetSize(
      int B,
      int L,
      int maxSize,
      const int* target,
      int* targetSize,
      cudaStream_t stream);

  /**
   * B: batch size
   * T: input length
   * N: dictionary size
   * scaleMode: type of size scaling
   * targetSize: [B] target sizes
   * scale: [B] (out) scale factor
   * stream: CUDA stream
   */
  static void computeScale(
      int B,
      int T,
      int N,
      CriterionScaleMode scaleMode,
      const int* targetSize,
      Float* scale,
      cudaStream_t stream);
};

} // namespace cuda
} // namespace w2l
