/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Utilities which may depend on ArrayFire / flashlight.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>

#include "common/Utils.h"

namespace w2l {

template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
  return vec;
}

template <typename T>
std::vector<T> afToVector(const fl::Variable& var) {
  return afToVector<T>(var.array());
}

/**
 * Reads a 2D padded character matrix from an `af::array` into a `std::vector`
 * of strings. Each column is read until the first `terminator` and then
 * converted to an `std::string`.
 * - `T` must correspond to `arr.type()`
 * - matrix values (except terminator) must be representable as `char`
 */
template <class T>
std::vector<std::string> afMatrixToStrings(const af::array& arr, T terminator) {
  int L = arr.dims(0); // padded length of string
  int N = arr.dims(1); // number of strings
  std::vector<std::string> result;
  auto values = afToVector<T>(arr);
  for (int i = 0; i < N; ++i) {
    const T* row = &values[i * L];
    int len = 0;
    while (len < L && row[len] != terminator) {
      ++len;
    }
    result.emplace_back(row, row + len);
  }
  return result;
}

int64_t numTotalParams(std::shared_ptr<fl::Module> module);

af::array
pad(const af::array& in, const int size, const int dim = 0, float val = 0.0);

} // namespace w2l
