/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

class WindowBase {
 public:
  WindowBase() {}

  virtual fl::Variable computeSingleStepWindow(
      const fl::Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) = 0;

  virtual fl::Variable
  computeWindowMask(int targetLen, int inputSteps, int batchSize) = 0;

  virtual ~WindowBase() {}

  void setBatchStat(int seqLen, int targetLen, int batchSize) {
    inputLen_ = seqLen;
    targetLen_ = targetLen;
    batchSize_ = batchSize;
  }

 protected:
  int inputLen_;
  int targetLen_;
  int batchSize_;

 private:
  FL_SAVE_LOAD()
};

} // namespace w2l
