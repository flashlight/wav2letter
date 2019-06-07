/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Derivatives.h"

#include <stdexcept>

#include "SpeechUtils.h"

namespace speech {

template <typename T>
Derivatives<T>::Derivatives(int64_t deltawindow, int64_t accwindow)
    : deltaWindow_(deltawindow), accWindow_(accwindow) {}

template <typename T>
std::vector<T> Derivatives<T>::apply(
    const std::vector<T>& input,
    int64_t numfeat) const {
  if (input.size() % numfeat != 0) {
    throw std::invalid_argument(
        "Derivatives: input size is not divisible by numFeatures");
  }
  // Compute deltas
  if (deltaWindow_ <= 0) {
    return input;
  }

  auto deltas = computeDerivative(input, deltaWindow_, numfeat);
  size_t szMul = 2;
  std::vector<T> doubledeltas;
  if (accWindow_ > 0) {
    // Compute double deltas (only if required)
    szMul = 3;
    doubledeltas = computeDerivative(deltas, accWindow_, numfeat);
  }
  std::vector<T> output(input.size() * szMul);
  int64_t numframes = input.size() / numfeat;
  for (size_t i = 0; i < numframes; ++i) {
    size_t curInIdx = i * numfeat;
    size_t curOutIdx = curInIdx * szMul;
    // copy input
    std::copy(
        input.data() + curInIdx,
        input.data() + curInIdx + numfeat,
        output.data() + curOutIdx);
    // copy deltas
    std::copy(
        deltas.data() + curInIdx,
        deltas.data() + curInIdx + numfeat,
        output.data() + curOutIdx + numfeat);
    // copy double-deltas
    if (accWindow_ > 0) {
      std::copy(
          doubledeltas.data() + curInIdx,
          doubledeltas.data() + curInIdx + numfeat,
          output.data() + curOutIdx + 2 * numfeat);
    }
  }
  return output;
}

template <typename T>
std::vector<T> Derivatives<T>::computeDerivative(
    const std::vector<T>& input,
    int64_t windowlen,
    int64_t numfeat) const {
  int64_t numframes = input.size() / numfeat;
  std::vector<T> output(input.size(), 0.0);
  T denominator = (windowlen * (windowlen + 1) * (2 * windowlen + 1)) / 3.0;
  for (size_t i = 0; i < numframes; ++i) {
    for (size_t j = 0; j < numfeat; ++j) {
      size_t curIdx = i * numfeat + j;
      for (size_t d = 1; d <= windowlen; ++d) {
        output[curIdx] += d *
            (input[curIdx + std::min((numframes - i - 1), d) * numfeat] -
             input[curIdx - std::min(i, d) * numfeat]);
      }
      output[curIdx] /= denominator;
    }
  }
  return output;
}

template class Derivatives<float>;
template class Derivatives<double>;
} // namespace speech
