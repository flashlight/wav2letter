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
template <typename T>
class PreEmphasis {
 public:
  PreEmphasis(T alpha, int64_t N);

  std::vector<T> apply(const std::vector<T>& input) const;

  void applyInPlace(std::vector<T>& input) const;

 private:
  T preemCoef_;
  int64_t windowLength_;
};
} // namespace w2l
