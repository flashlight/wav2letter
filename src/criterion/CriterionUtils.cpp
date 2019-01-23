/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CriterionUtils.h"

#include <algorithm>
#include <cmath>

using fl::Variable;

namespace w2l {

int countRepeats(const int* labels, int len) {
  int r = 0;
  for (int i = 1; i < len; ++i) {
    if (labels[i] == labels[i - 1]) {
      ++r;
    }
  }
  return r;
}

int getTargetSize(const int* labels, int len) {
  while (len > 0 && labels[len - 1] < 0) {
    --len;
  }
  return len;
}

CriterionScaleMode getCriterionScaleMode(
    const std::string& onorm,
    bool sqnorm) {
  if (onorm == "none") {
    return CriterionScaleMode::NONE;
  } else if (onorm == "input") {
    return sqnorm ? CriterionScaleMode::INPUT_SZ_SQRT
                  : CriterionScaleMode::INPUT_SZ;
  } else if (onorm == "target") {
    return sqnorm ? CriterionScaleMode::TARGET_SZ_SQRT
                  : CriterionScaleMode::TARGET_SZ;
  } else {
    throw std::invalid_argument("invalid onorm option");
  }
}

CriterionScaleFn getCriterionScaleFn(CriterionScaleMode scale) {
  switch (scale) {
    case CriterionScaleMode::NONE:
      return
          [](int64_t /* unused */, int64_t /* unused */, int64_t /* unused */) {
            return 1.0;
          };
    case CriterionScaleMode::INPUT_SZ:
      return [](int64_t /* unused */, int64_t T, int64_t /* unused */) {
        return (T > 0) ? (1.0 / T) : 1.0;
      };
    case CriterionScaleMode::INPUT_SZ_SQRT:
      return [](int64_t /* unused */, int64_t T, int64_t /* unused */) {
        return (T > 0) ? std::sqrt(1.0 / T) : 1.0;
      };
    case CriterionScaleMode::TARGET_SZ:
      return [](int64_t /* unused */, int64_t /* unused */, int64_t L) {
        return (L > 0) ? (1.0 / L) : 1.0;
      };
    case CriterionScaleMode::TARGET_SZ_SQRT:
      return [](int64_t /* unused */, int64_t /* unused */, int64_t L) {
        return (L > 0) ? std::sqrt(1.0 / L) : 1.0;
      };
    default:
      throw(af::exception("Unsupported criterion scale mode!"));
      return
          [](int64_t /* unused */, int64_t /* unused */, int64_t /* unused */) {
            return 1.0;
          };
      // return so that compiler doesn't compain
  }
}
Variable getLinearTarget(const Variable& targetVar, int T) {
  int L = targetVar.dims(0);
  int B = targetVar.dims(1);

  std::vector<int> target(B * L);
  std::vector<int> newTarget(B * T);

  targetVar.host(target.data());
  for (int b = 0; b < B; ++b) {
    const auto pTarget = target.data() + b * L;
    auto pNewTarget = newTarget.data() + b * T;

    int TN = w2l::getTargetSize(pTarget, L);
    if (TN > T || TN == 0) {
      // hacky way to indicate that LinSeg should output NAN,
      // make ASG think TN == 0.
      std::fill(pNewTarget, pNewTarget + T, -1);
    } else {
      for (int t = 0; t < T; ++t) {
        pNewTarget[t] = pTarget[t * TN / T];
      }
    }
  }
  return Variable(af::array(T, B, newTarget.data()), false);
}

} // namespace w2l
