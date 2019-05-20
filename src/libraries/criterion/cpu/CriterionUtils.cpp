/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cpu/CriterionUtils.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace w2l {
namespace cpu {

template <class Float>
void CriterionUtils<Float>::batchTargetSize(
    int B,
    int L,
    int maxSize,
    const int* target,
    int* targetSize) {
  for (int b = 0; b < B; ++b) {
    for (int i = L - 1; i >= 0; --i) {
      if (target[b * L + i] >= 0) {
        targetSize[b] = std::min(maxSize, i + 1);
        break;
      }
    }
  }
}

template <class Float>
void CriterionUtils<Float>::computeScale(
    int B,
    int T,
    int /* N */,
    CriterionScaleMode scaleMode,
    const int* targetSize,
    Float* scale) {
  for (int b = 0; b < B; ++b) {
    switch (scaleMode) {
      case CriterionScaleMode::NONE:
        scale[b] = 1.0;
        break;
      case CriterionScaleMode::INPUT_SZ:
        scale[b] = T > 0 ? 1.0 / T : 1.0;
        break;
      case CriterionScaleMode::INPUT_SZ_SQRT:
        scale[b] = T > 0 ? std::sqrt(1.0 / T) : 1.0;
        break;
      case CriterionScaleMode::TARGET_SZ:
        scale[b] = targetSize[b] > 0 ? 1.0 / targetSize[b] : 1.0;
        break;
      case CriterionScaleMode::TARGET_SZ_SQRT:
        scale[b] = targetSize[b] > 0 ? std::sqrt(1.0 / targetSize[b]) : 1.0;
        break;
      default:
        break;
    }
  }
}

template struct CriterionUtils<float>;
template struct CriterionUtils<double>;

} // namespace cpu
} // namespace w2l
