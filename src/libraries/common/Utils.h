/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

namespace w2l {

/// Zeroes `count * sizeof(T)` bytes
template <typename T>
void setZero(T* ptr, size_t count) {
  std::memset(ptr, 0, count * sizeof(T));
}

} // namespace w2l
