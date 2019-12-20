/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Ceplifter.h"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace w2l {

Ceplifter::Ceplifter(int numfilters, int lifterparam)
    : numFilters_(numfilters), lifterParam_(lifterparam), coefs_(numFilters_) {
  std::iota(coefs_.begin(), coefs_.end(), 0.0);
  for (auto& c : coefs_) {
    c = 1.0 + 0.5 * lifterParam_ * std::sin(M_PI * c / lifterParam_);
  }
}

std::vector<float> Ceplifter::apply(const std::vector<float>& input) const {
  auto output(input);
  applyInPlace(output);
  return output;
}

void Ceplifter::applyInPlace(std::vector<float>& input) const {
  if (input.size() % numFilters_ != 0) {
    throw std::invalid_argument(
        "Ceplifter: input size is not divisible by numFilters");
  }
  size_t n = 0;
  for (auto& in : input) {
    in *= coefs_[n++];
    if (n == numFilters_) {
      n = 0;
    }
  }
}
} // namespace w2l
