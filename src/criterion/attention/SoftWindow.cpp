/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SoftWindow.h"

using namespace fl;

namespace w2l {

SoftWindow::SoftWindow() {}
SoftWindow::SoftWindow(double std, double avgRate, int offset)
    : std_(std), avgRate_(avgRate), offset_(offset) {}

int SoftWindow::getCenter(int step, int inputSteps) {
  return static_cast<int>(
      std::round(std::min(offset_ + step * avgRate_, inputSteps - avgRate_)));
}

Variable SoftWindow::computeSingleStepWindow(
    const Variable& /* unused */,
    int inputSteps,
    int batchSize,
    int step) {
  int cidx = getCenter(step, inputSteps);

  auto maskArray = af::range(af::dim4(inputSteps));
  maskArray = exp(-pow(maskArray - cidx, 2) / (2 * std_ * std_));

  // [1, inputSteps, batchSize]
  auto mask = Variable(
      tile(moddims(maskArray, {1, inputSteps}), {1, 1, batchSize}), false);

  return mask;
}

Variable
SoftWindow::computeWindowMask(int targetLen, int inputSteps, int batchSize) {
  std::vector<int> centerVec(targetLen, 0);
  for (int u = 0; u < targetLen; ++u) {
    centerVec[u] = getCenter(u, inputSteps);
  }

  auto ts = af::range(af::dim4(inputSteps));
  ts = af::tile(af::moddims(ts, {1, inputSteps}), {targetLen, 1});
  auto centers =
      af::tile(af::array(targetLen, 1, centerVec.data()), {1, inputSteps});
  auto maskArray = exp(-pow(ts - centers, 2) / (2 * std_ * std_));

  // [targetLen, inputSteps, batchSize]
  auto mask = Variable(tile(maskArray, {1, 1, batchSize}), false);

  return mask;
}

} // namespace w2l
