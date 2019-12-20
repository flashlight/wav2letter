/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <vector>

template <typename T>
bool compareVec(std::vector<T> A, std::vector<T> B, float precision = 1E-5) {
  if (A.size() != B.size()) {
    return false;
  }
  for (std::size_t i = 0; i < A.size(); ++i) {
    if (std::abs(A[i] - B[i]) > precision) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::vector<T> randVec(std::size_t N, float min = -1.0, float max = 1.0) {
  std::vector<T> vec(N);
  for (auto& v : vec) {
    v = static_cast<T>(rand()) / static_cast<T>(RAND_MAX);
    v = v * (max - min) + min;
  }
  return vec;
}

template <typename T>
std::vector<T> transposeVec(const std::vector<T>& in, int inRow, int inCol) {
  std::vector<T> out(inRow * inCol);
  for (size_t r = 0; r < inRow; ++r) {
    for (size_t c = 0; c < inCol; ++c) {
      out[c * inRow + r] = in[r * inCol + c];
    }
  }
  return out;
}
