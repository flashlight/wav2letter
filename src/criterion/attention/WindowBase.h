/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace fl {

class WindowBase {
 public:
  WindowBase() {}

  virtual Variable computeSingleStepWindow(
      const Variable& prevAttn,
      int inputSteps,
      int batchSize,
      int step) = 0;

  virtual Variable
  computeWindowMask(int targetLen, int inputSteps, int batchSize) = 0;

  virtual ~WindowBase() {}

 private:
  FL_SAVE_LOAD()
};

} // namespace fl
