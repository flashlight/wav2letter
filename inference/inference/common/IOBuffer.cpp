/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/common/IOBuffer.h"

#include <math.h>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

IOBuffer::IOBuffer(int initialSize, const std::string& name)
    : buf_(initialSize), offsetInBytes_(0), sizeInBytes_(0), name_(name) {}

IOBuffer::IOBuffer(const void* buffer, int sizeInBytes, const std::string& name)
    : buf_(
          static_cast<const char*>(buffer),
          static_cast<const char*>(buffer) + sizeInBytes),
      offsetInBytes_(0),
      sizeInBytes_(sizeInBytes),
      name_(name) {
  if (sizeInBytes < 0) {
    std::stringstream ss;
    ss << "Invalid index at IOBuffer::IOBuffer(buffer=" << buffer
       << " sizeInBytes=" << sizeInBytes << " name=" << name << ")";
    throw std::invalid_argument(ss.str());
  }
}

int IOBuffer::headRoom() const {
  if (offsetInBytes_ < 0) {
    throw std::runtime_error(
        "Invalid offsetInBytes_=" + std::to_string(offsetInBytes_) +
        " in IOBuffer::headRoom()");
  }
  return offsetInBytes_;
}

int IOBuffer::tailRoom() const {
  return buf_.size() - headRoom() - sizeInBytes_;
}

void IOBuffer::reset() {
  if (buf_.data()) {
    std::memmove(buf_.data(), data<void>(), sizeInBytes_);
  }
  offsetInBytes_ = 0;
}

std::string IOBuffer::debugString() const {
  return debugStringHelper();
}

std::string IOBuffer::debugStringHelper(const std::string& content) const {
  std::stringstream ss;
  ss << "IOBuffer:{"
        "name_="
     << name_ << " offsetInBytes_=" << offsetInBytes_
     << " buf_.size()=" << buf_.size() << " sizeInBytes_=" << sizeInBytes_;
  if (!content.empty()) {
    ss << " content:" << content;
  }
  ss << "}";
  return ss.str();
}

void IOBuffer::clear() {
  offsetInBytes_ = 0;
  sizeInBytes_ = 0;
  buf_.clear();
}

std::vector<char>& IOBuffer::buffer() {
  return buf_;
}

} // namespace streaming
} // namespace w2l
