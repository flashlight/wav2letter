/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MedianWindow.h"

using namespace fl;

namespace w2l {

MedianWindow::MedianWindow() {}
MedianWindow::MedianWindow(int wL, int wR) : wL_(wL), wR_(wR) {}

Variable MedianWindow::initialize(int inputSteps, int batchSize) {
  int width = std::min(wL_ + wR_, inputSteps);

  // [1, inputSteps]
  auto maskArray = af::constant(0.0, 1, inputSteps);
  maskArray(af::span, af::seq(0, width - 1)) = 1.0;

  // [1, inputSteps, batchSize]
  auto mask = Variable(tile(maskArray, {1, 1, batchSize}), false);

  return mask;
}

Variable MedianWindow::computeSingleStepWindow(
    const Variable& prevAttn, // [1, windowsize, batchSize]
    int inputSteps,
    int batchSize,
    int step) {
  int width = std::min(wL_ + wR_, inputSteps);

  if (step == 0 || width >= inputSteps) {
    return initialize(inputSteps, batchSize);
  }
  // Each row of prevAttn is the attention for an input utterance.
  // The attention vector is output from a softmax.
  // The definition of "median" is the point where cdf passes 0.5.
  auto mIdx = sum(accum(prevAttn.array(), 1) < 0.5, 1).as(af::dtype::s32);
  auto startIdx = mIdx - wL_;

  // check boundary conditions and adjust the window
  auto startDiff = af::abs(clamp(startIdx, -wL_, 0));
  startIdx = startIdx + startDiff;
  auto endDiff =
      af::abs(clamp(startIdx + wL_ + wR_ - inputSteps, 0, wL_ + wR_));
  startIdx = startIdx - endDiff;

  auto maskArray = af::constant(0.0, 1, inputSteps, batchSize, f32);
  auto indices = range(af::dim4(width, batchSize), 0) +
      tile(moddims(startIdx, {1, batchSize}), {width, 1}) +
      tile(moddims(
               af::seq(0, batchSize * inputSteps - 1, inputSteps),
               {1, batchSize}),
           {width, 1});
  maskArray(flat(indices)) = 1.0;

  // [1, inputSteps, batchSize]
  auto mask = Variable(maskArray, false);

  return mask;
}

Variable MedianWindow::computeWindowMask(
    int /* unused */,
    int /* unused */,
    int /* unused */) {
  throw af::exception("MedianWindow does not support vectorized window mask");
}

} // namespace w2l
