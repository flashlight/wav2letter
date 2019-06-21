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

// Compute first order (deltas) and second order (acceleration) derivatives of
//  cepstral coefficients
//    d(i) =    0.5 * SUM_t (t * (c(i + t) - c (i - t))) / SUM_t t^2
//      where t in [1, maxlagsize]

template <typename T>
class Derivatives {
 public:
  Derivatives(int64_t deltawindow, int64_t accwindow);

  std::vector<T> apply(const std::vector<T>& input, int64_t numfeat) const;

 private:
  int64_t deltaWindow_; // delta derivatives lag size
  int64_t accWindow_; // acceleration derivatives lag size

  // Helper function to compute derivatives of single order
  std::vector<T> computeDerivative(
      const std::vector<T>& input,
      int64_t windowlen,
      int64_t numfeat) const;
};
} // namespace w2l
