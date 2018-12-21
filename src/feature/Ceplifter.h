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

namespace speech {

// Re-scale the cepstral coefficients using liftering
//    c'(n) = c(n) * (1 + 0.5 * L * sin(pi * n/ L)) where L is lifterparam
template <typename T>
class Ceplifter {
 public:
  Ceplifter(int64_t numfilters, int64_t lifterparam);

  std::vector<T> apply(const std::vector<T>& input) const;

  void applyInPlace(std::vector<T>& input) const;

 private:
  int64_t numFilters_; // number of filterbank channels
  int64_t lifterParam_; // liftering parameter
  std::vector<T> coefs_; // coefficients to scale cepstral coefficients
};
} // namespace speech
