/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

#include "FeatureParams.h"

namespace w2l {

// Applies a given window on input
//    s'(n) = w(n) * s(n) where w(n) are the window coefficients

class Windowing {
 public:
  Windowing(int N, WindowType window);

  std::vector<float> apply(const std::vector<float>& input) const;

  void applyInPlace(std::vector<float>& input) const;

 private:
  int windowLength_;
  WindowType windowType_;
  std::vector<float> coefs_;
};
} // namespace w2l
