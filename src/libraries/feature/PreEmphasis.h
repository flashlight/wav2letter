/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

namespace w2l {

// Pre-emphasise the signal by applying the first order difference equation
//    s'(n) = s(n) - k * s(n-1)  where k in [0, 1)

class PreEmphasis {
 public:
  PreEmphasis(float alpha, int N);

  std::vector<float> apply(const std::vector<float>& input) const;

  void applyInPlace(std::vector<float>& input) const;

 private:
  float preemCoef_;
  int windowLength_;
};
} // namespace w2l
