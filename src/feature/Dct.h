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
template <typename T>
class Dct {
 public:
  Dct(int64_t numfilters, int64_t numceps);

  std::vector<T> apply(const std::vector<T>& input) const;

 private:
  int64_t numFilters_; // Number of filterbank channels
  int64_t numCeps_; // Number of cepstral coefficients
  std::vector<T> dctMat_; // Dct matrix
};
} // namespace w2l
