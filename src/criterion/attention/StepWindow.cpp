/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "StepWindow.h"

using namespace fl;

namespace w2l {

StepWindow::StepWindow() {}
StepWindow::StepWindow(int sMin, int sMax, double vMin, double vMax)
    : sMin_(sMin), sMax_(sMax), vMin_(vMin), vMax_(vMax) {}

Variable StepWindow::computeSingleStepWindow(
    const Variable& /* unused */,
    int inputSteps,
    int batchSize,
    int step) {
  int start_idx = std::max(
      0,
      static_cast<int>(
          std::round(std::min(inputSteps - vMax_, sMin_ + step * vMin_))));
  int end_idx =
      std::min(static_cast<int>(std::round(sMax_ + step * vMax_)), inputSteps);

  std::vector<float> maskvec(inputSteps, 0.0);
  std::fill(maskvec.begin() + start_idx, maskvec.begin() + end_idx, 1.0);

  // [1, inputSteps]
  auto maskarray = af::constant(0.0, 1, inputSteps);
  maskarray(af::span, af::seq(start_idx, end_idx - 1)) = 1.0;

  // [1, inputSteps, batchSize]
  auto mask = Variable(tile(maskarray, {1, 1, batchSize}), false);

  return mask;
}

Variable
StepWindow::computeWindowMask(int targetLen, int inputSteps, int batchSize) {
  std::vector<Variable> maskvec(targetLen, Variable());
  for (int u = 0; u < targetLen; ++u) {
    maskvec[u] = computeSingleStepWindow(
        Variable() /* unused */, inputSteps, batchSize, u);
  }

  // [targetLen, inputSteps, batchSize]
  auto mask = concatenate(maskvec, 0);

  return mask;
}

} // namespace w2l
