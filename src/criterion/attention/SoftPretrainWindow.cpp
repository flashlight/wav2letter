/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/attention/SoftPretrainWindow.h"

using namespace fl;

namespace w2l {

SoftPretrainWindow::SoftPretrainWindow(double std) : std_(std) {}

/* The pretrain window should only be used during training, since it requires
 * users set the length of the targets using setBatchStat() in advance.*/
Variable SoftPretrainWindow::computeSingleStepWindow(
    const Variable& /* unused */,
    int inputSteps,
    int batchSize,
    int step) {
  auto ts = af::range(af::dim4(1, inputSteps), 1);
  double vratio = (double)inputSteps / (double)targetLen_;
  auto maskArray = exp(-pow(ts - vratio * step, 2) / (2 * std_ * std_));

  // [1, inputSteps, batchSize]
  return Variable(tile(maskArray, {1, 1, batchSize}), false);
}

Variable SoftPretrainWindow::computeWindowMask(
    int targetLen,
    int inputSteps,
    int batchSize) {
  auto ts = af::range(af::dim4(targetLen, inputSteps), 1);
  auto us = af::range(af::dim4(targetLen, inputSteps));
  double vratio = (double)inputSteps / (double)targetLen;
  auto maskArray = exp(-pow(ts - vratio * us, 2) / (2 * std_ * std_));

  // [targetLen, inputSteps, batchSize]
  return Variable(tile(maskArray, {1, 1, batchSize}), false);
}

} // namespace w2l
