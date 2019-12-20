/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PreEmphasis.h"

#include <cstddef>
#include <stdexcept>

namespace w2l {

PreEmphasis::PreEmphasis(float alpha, int N)
    : preemCoef_(alpha), windowLength_(N) {
  if (windowLength_ <= 1) {
    throw std::invalid_argument("PreEmphasis: windowLength must be > 1");
  }
  if (preemCoef_ < 0.0 || preemCoef_ >= 1.0) {
    throw std::invalid_argument("PreEmphasis: alpha must be in [0, 1)");
  }
};

std::vector<float> PreEmphasis::apply(const std::vector<float>& input) const {
  auto output(input);
  applyInPlace(output);
  return output;
}

void PreEmphasis::applyInPlace(std::vector<float>& input) const {
  if (input.size() % windowLength_ != 0) {
    throw std::invalid_argument(
        "PreEmphasis: input.size() not divisible by windowLength");
  }
  size_t nframes = input.size() / windowLength_;
  for (size_t n = nframes; n > 0; --n) {
    size_t e = n * windowLength_ - 1; // end of current frame
    size_t s = (n - 1) * windowLength_; // start of current frame
    for (size_t i = e; i > s; --i) {
      input[i] -= (preemCoef_ * input[i - 1]);
    }
    input[s] *= (1 - preemCoef_);
  }
}
} // namespace w2l
