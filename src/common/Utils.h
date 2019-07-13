/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <arrayfire.h>
#include <flashlight/flashlight.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Utils-base.h"

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

/**
 * Calls `f(args...)` repeatedly, retrying if an exception is thrown.
 * Supports sleeps between retries, with duration starting at `initial` and
 * multiplying by `factor` each retry. At most `maxIters` calls are made.
 */
template <class Fn, class... Args>
fl::cpp::result_of_t<Fn(Args...)> retryWithBackoff(
    std::chrono::duration<double> initial,
    double factor,
    int64_t maxIters,
    Fn&& f,
    Args&&... args);

template <
    typename FwdIt,
    typename = fl::cpp::enable_if_t<std::is_same<
        fl::cpp::decay_t<decltype(*std::declval<FwdIt>())>,
        std::string>::value>>
std::string join(const std::string& delim, FwdIt begin, FwdIt end);

std::string join(const std::string& delim, const std::vector<std::string>& vec);

template <class... Args>
std::string format(const char* fmt, Args&&... args);

int64_t numTotalParams(std::shared_ptr<fl::Module> module);

} // namespace w2l

#include "Utils-inl.h"
