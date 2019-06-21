/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Dct.h"

#include <cmath>
#include <numeric>

#include "SpeechUtils.h"

namespace speech {

template <typename T>
Dct<T>::Dct(int64_t numfilters, int64_t numceps)
    : numFilters_(numfilters),
      numCeps_(numceps),
      dctMat_(numfilters * numceps) {
  for (size_t f = 0; f < numFilters_; ++f) {
    for (size_t c = 0; c < numCeps_; ++c) {
      dctMat_[f * numCeps_ + c] = std::sqrt(2.0 / numFilters_) *
          std::cos(M_PI * c * (f + 0.5) / numFilters_);
    }
  }
}

template <typename T>
std::vector<T> Dct<T>::apply(const std::vector<T>& input) const {
  return cblasGemm(input, dctMat_, numCeps_, numFilters_);
}

template class Dct<float>;
template class Dct<double>;
} // namespace speech
