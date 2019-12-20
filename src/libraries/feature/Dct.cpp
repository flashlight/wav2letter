/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Dct.h"

#include <cmath>
#include <cstddef>
#include <numeric>

#include "SpeechUtils.h"

namespace w2l {

Dct::Dct(int numfilters, int numceps)
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

std::vector<float> Dct::apply(const std::vector<float>& input) const {
  return cblasGemm(input, dctMat_, numCeps_, numFilters_);
}
} // namespace w2l
