/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "PreEmphasis.h"

#include <glog/logging.h>

namespace speech {

template <typename T>
PreEmphasis<T>::PreEmphasis(T alpha, int64_t N)
    : preemCoef_(alpha), windowLength_(N) {
  LOG_IF(FATAL, windowLength_ < 2);
  LOG_IF(FATAL, preemCoef_ < 0.0 || preemCoef_ >= 1.0);
};

template <typename T>
std::vector<T> PreEmphasis<T>::apply(const std::vector<T>& input) const {
  auto output(input);
  applyInPlace(output);
  return output;
}

template <typename T>
void PreEmphasis<T>::applyInPlace(std::vector<T>& input) const {
  LOG_IF(FATAL, (input.size() % windowLength_) != 0);
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

template class PreEmphasis<float>;
template class PreEmphasis<double>;
} // namespace speech
