/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>

namespace w2l {

template <typename FwdIt, typename>
std::string join(const std::string& delim, FwdIt begin, FwdIt end) {
  if (begin == end) {
    return "";
  }

  size_t totalSize = begin->size();
  for (auto it = std::next(begin); it != end; ++it) {
    totalSize += delim.size() + it->size();
  }

  std::string result;
  result.reserve(totalSize);

  result.append(*begin);
  for (auto it = std::next(begin); it != end; ++it) {
    result.append(delim);
    result.append(*it);
  }
  return result;
}

template <class... Args>
std::string format(const char* fmt, Args&&... args) {
  auto res = std::snprintf(nullptr, 0, fmt, std::forward<Args>(args)...);
  if (res < 0) {
    throw std::runtime_error(std::strerror(errno));
  }
  std::string buf(res, '\0');
  // the size here is fine -- it's legal to write '\0' to buf[res]
  auto res2 = std::snprintf(&buf[0], res + 1, fmt, std::forward<Args>(args)...);
  if (res2 < 0) {
    throw std::runtime_error(std::strerror(errno));
  }

  if (res2 != res) {
    throw std::runtime_error(
        "The size of the formated string is not equal to what it is expected.");
  }
  return buf;
}

template <class Fn, class... Args>
typename std::result_of<Fn(Args...)>::type retryWithBackoff(
    std::chrono::duration<double> initial,
    double factor,
    int64_t maxIters,
    Fn&& f,
    Args&&... args) {
  if (!(initial.count() >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad initial");
  } else if (!(factor >= 0.0)) {
    throw std::invalid_argument("retryWithBackoff: bad factor");
  } else if (maxIters <= 0) {
    throw std::invalid_argument("retryWithBackoff: bad maxIters");
  }
  auto sleepSecs = initial.count();
  for (int64_t i = 0; i < maxIters; ++i) {
    try {
      return f(std::forward<Args>(args)...);
    } catch (...) {
      if (i >= maxIters - 1) {
        throw;
      }
    }
    if (sleepSecs > 0.0) {
      /* sleep override */
      std::this_thread::sleep_for(
          std::chrono::duration<double>(std::min(1e7, sleepSecs)));
    }
    sleepSecs *= factor;
  }
  throw std::logic_error("retryWithBackoff: hit unreachable");
}

} // namespace w2l
