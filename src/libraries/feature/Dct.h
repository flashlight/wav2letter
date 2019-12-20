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

// Compute Discrete Cosine Transform
//    c(i) = sqrt(2/N)  SUM_j (m(j) * cos(pi * i * (j - 0.5)/ N))
//      where j in [1, N], m - log filterbank amplitudes

class Dct {
 public:
  Dct(int numfilters, int numceps);

  std::vector<float> apply(const std::vector<float>& input) const;

 private:
  int numFilters_; // Number of filterbank channels
  int numCeps_; // Number of cepstral coefficients
  std::vector<float> dctMat_; // Dct matrix
};
} // namespace w2l
