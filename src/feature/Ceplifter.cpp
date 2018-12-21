/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Ceplifter.h"

#include <cmath>
#include <numeric>

#include <glog/logging.h>

namespace speech {

template <typename T>
Ceplifter<T>::Ceplifter(int64_t numfilters, int64_t lifterparam)
    : numFilters_(numfilters), lifterParam_(lifterparam), coefs_(numFilters_) {
  std::iota(coefs_.begin(), coefs_.end(), 0.0);
  for (auto& c : coefs_) {
    c = 1.0 + 0.5 * lifterParam_ * std::sin(M_PI * c / lifterParam_);
  }
}

template <typename T>
std::vector<T> Ceplifter<T>::apply(const std::vector<T>& input) const {
  auto output(input);
  applyInPlace(output);
  return output;
}

template <typename T>
void Ceplifter<T>::applyInPlace(std::vector<T>& input) const {
  LOG_IF(FATAL, (input.size() % numFilters_) != 0);
  size_t n = 0;
  for (auto& in : input) {
    in *= coefs_[n++];
    if (n == numFilters_) {
      n = 0;
    }
  }
}

template class Ceplifter<float>;
template class Ceplifter<double>;
} // namespace speech
