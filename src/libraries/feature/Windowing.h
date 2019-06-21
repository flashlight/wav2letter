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
template <typename T>
class Windowing {
 public:
  Windowing(int64_t N, WindowType window);

  std::vector<T> apply(const std::vector<T>& input) const;

  void applyInPlace(std::vector<T>& input) const;

 private:
  int64_t windowLength_;
  WindowType windowType_;
  std::vector<T> coefs_;
};
} // namespace w2l
