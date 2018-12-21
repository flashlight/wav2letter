/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Dither.h"

#include <time.h>

namespace speech {

template <typename T>
Dither<T>::Dither(T ditherVal)
    : ditherVal_(ditherVal), rng_((ditherVal > 0.0) ? 123456 : time(nullptr)){};

template <typename T>
std::vector<T> Dither<T>::apply(const std::vector<T>& input) {
  auto output(input);
  applyInPlace(output);
  return output;
}

template <typename T>
void Dither<T>::applyInPlace(std::vector<T>& input) {
  std::uniform_real_distribution<T> distribution(0.0, 1.0);
  for (auto& i : input) {
    i += ditherVal_ * distribution(rng_);
  }
}

template class Dither<float>;
template class Dither<double>;
} // namespace speech
