/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "WindowBase.h"

namespace fl {

class MedianWindow : public WindowBase {
 public:
  MedianWindow();
  MedianWindow(int wL, int wR);

  Variable initialize(int inputSteps, int batchSize);

  Variable computeSingleStepWindow(
      const Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) override;

  Variable computeWindowMask(int targetLen, int inputSteps, int batchSize)
      override;

 private:
  int wL_;
  int wR_;

  FL_SAVE_LOAD_WITH_BASE(WindowBase, wL_, wR_)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::MedianWindow)
