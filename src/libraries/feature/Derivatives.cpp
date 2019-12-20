/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Derivatives.h"

#include <cstddef>
#include <stdexcept>

#include "SpeechUtils.h"

namespace w2l {

Derivatives::Derivatives(int deltawindow, int accwindow)
    : deltaWindow_(deltawindow), accWindow_(accwindow) {}

std::vector<float> Derivatives::apply(
    const std::vector<float>& input,
    int numfeat) const {
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
  std::vector<float> doubledeltas;
  if (accWindow_ > 0) {
    // Compute double deltas (only if required)
    szMul = 3;
    doubledeltas = computeDerivative(deltas, accWindow_, numfeat);
  }
  std::vector<float> output(input.size() * szMul);
  int numframes = input.size() / numfeat;
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

std::vector<float> Derivatives::computeDerivative(
    const std::vector<float>& input,
    int windowlen,
    int numfeat) const {
  int numframes = input.size() / numfeat;
  std::vector<float> output(input.size(), 0.0);
  float denominator = (windowlen * (windowlen + 1) * (2 * windowlen + 1)) / 3.0;
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

} // namespace w2l
