/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Windowing.h"

#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>

namespace w2l {

Windowing::Windowing(int N, WindowType windowtype)
    : windowLength_(N), windowType_(windowtype), coefs_(N) {
  if (windowLength_ <= 1) {
    throw std::invalid_argument("Windowing: windowLength must be > 1");
  }
  std::iota(coefs_.begin(), coefs_.end(), 0.0);
  switch (windowtype) {
    case WindowType::HAMMING:
      for (auto& c : coefs_) {
        c = 0.54 - 0.46 * std::cos(2 * M_PI * c / (N - 1));
      }
      break;
    case WindowType::HANNING:
      for (auto& c : coefs_) {
        c = 0.5 * (1.0 - std::cos(2 * M_PI * c / (N - 1)));
      }
      break;
    default:
      throw std::invalid_argument("Windowing: unsupported window type");
  }
}

std::vector<float> Windowing::apply(const std::vector<float>& input) const {
  auto output(input);
  applyInPlace(output);
  return output;
}

void Windowing::applyInPlace(std::vector<float>& input) const {
  if (input.size() % windowLength_ != 0) {
    throw std::invalid_argument(
        "Windowing: input size is not divisible by windowLength");
  }
  size_t n = 0;
  for (auto& in : input) {
    in *= coefs_[n++];
    if (n == windowLength_) {
      n = 0;
    }
  }
}
} // namespace w2l
