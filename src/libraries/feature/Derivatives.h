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

class Derivatives {
 public:
  Derivatives(int deltawindow, int accwindow);

  std::vector<float> apply(const std::vector<float>& input, int numfeat) const;

 private:
  int deltaWindow_; // delta derivatives lag size
  int accWindow_; // acceleration derivatives lag size

  // Helper function to compute derivatives of single order
  std::vector<float> computeDerivative(
      const std::vector<float>& input,
      int windowlen,
      int numfeat) const;
};
} // namespace w2l
