/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <cstdlib>
#include <vector>

namespace w2l {
namespace streaming {
template <typename T>
std::vector<T> randVec(int size) {
  std::vector<T> vec(size);
  for (auto& i : vec) {
    i = static_cast<T>(static_cast<double>(rand()) / RAND_MAX);
  }
  return vec;
}
} // namespace streaming
} // namespace w2l
