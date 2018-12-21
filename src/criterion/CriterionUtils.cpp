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

int64_t countRepeats(const int* labels, int64_t len) {
  int64_t r = 0;
  for (int64_t i = 1; i < len; ++i) {
    if (labels[i] == labels[i - 1]) {
      ++r;
    }
  }
  return r;
}

int64_t getTargetSize(const int* labels, int64_t len) {
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

af::array viterbiPath(const af::array& input, const af::array& trans) {
  if (input.isempty()) {
    return af::array();
  }
  auto N = input.dims(0);
  auto T = input.dims(1);
  auto B = input.dims(2);
  std::vector<float> inputRaw(N * T * B);
  std::vector<float> transRaw(N * N);
  std::vector<int> res(T * B);

  input.host(inputRaw.data());
  trans.host(transRaw.data());

  std::vector<float> alpha(N * T);
  std::vector<int> beta(N * T);

  for (int b = 0; b < B; ++b) {
    std::copy(
        inputRaw.begin() + b * N * T,
        inputRaw.begin() + b * N * T + N,
        alpha.begin());

    for (int t = 1; t < T; t++) {
      float* alphaCurFrame = alpha.data() + t * N;
      float* alphaPrevFrame = alpha.data() + (t - 1) * N;
      float* inputCurFrame = inputRaw.data() + t * N + b * N * T;
      int* betaCurFrame = beta.data() + t * N;

      for (int i = 0; i < N; i++) {
        float max = NEG_INFINITY_FLT;
        for (int j = 0; j < N; j++) {
          float z = alphaPrevFrame[j] + transRaw[i * N + j];
          if (max < z) {
            betaCurFrame[i] = j;
            max = z;
          }
        }
        alphaCurFrame[i] = max + inputCurFrame[i];
      }
    }

    float max = NEG_INFINITY_FLT;
    float* alphaCurFrame = alpha.data() + (T - 1) * N;
    int pos = -1;
    for (int i = 0; i < N; i++) {
      if (max < alphaCurFrame[i]) {
        max = alphaCurFrame[i];
        pos = i;
      }
    }
    res[b * T + T - 1] = pos;
    for (int i = T - 1; i > 0; i--) {
      pos = beta[i * N + pos];
      res[b * T + i - 1] = pos;
    }
  }

  return af::array(T, B, res.data());
}

Variable getLinearTarget(const Variable& targetVar, intl T) {
  intl L = targetVar.dims(0);
  intl B = targetVar.dims(1);

  std::vector<int> target(B * L);
  std::vector<int> newTarget(B * T);

  targetVar.host(target.data());
  for (intl b = 0; b < B; ++b) {
    const auto pTarget = target.data() + b * L;
    auto pNewTarget = newTarget.data() + b * T;

    intl TN = w2l::getTargetSize(pTarget, L);
    if (TN > T || TN == 0) {
      // hacky way to indicate that LinSeg should output NAN,
      // make ASG think TN == 0.
      std::fill(pNewTarget, pNewTarget + T, -1);
    } else {
      for (intl t = 0; t < T; ++t) {
        pNewTarget[t] = pTarget[t * TN / T];
      }
    }
  }
  return Variable(af::array(T, B, newTarget.data()), false);
}

} // namespace w2l
