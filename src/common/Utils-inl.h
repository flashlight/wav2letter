/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cerrno>
#include <cstring>

namespace w2l {

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
  DCHECK_EQ(res2, res);
  return buf;
}

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

} // namespace w2l
