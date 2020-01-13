/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/vector.hpp>
#include <algorithm>
#include <string>
#include <vector>

namespace w2l {
namespace streaming {

// Inspired by folly::IOBuffer
// Currently memory is unalinged, will make it aligned if it improves
// performance.
class IOBuffer {
 public:
  explicit IOBuffer(int initialSize = 0, const std::string& name = "");

  IOBuffer(const void* buffer, int sizeInBytes, const std::string& name = "");

  template <typename T>
  T* data();

  template <typename T>
  const T* data() const;

  template <typename T>
  T* tail();

  template <typename T>
  const T* tail() const;

  template <typename T>
  void consume(int size);

  // Ensure that write() succeeds to write of buffer of given size. Allocate
  // larger buffer space if needed.
  template <typename T>
  void ensure(int size);

  template <typename T>
  void write(const T* buf, int size);

  template <typename T>
  void writeZero(int size);

  template <typename T>
  void move(int size);

  // Marks buffer as empty.
  void clear();

  template <typename T>
  int size() const;

  std::string debugString() const;

  template <typename T>
  std::string debugStringWithContent() const;

  std::vector<char>& buffer();

 private:
  int headRoom() const;
  int tailRoom() const;
  void reset();
  std::string debugStringHelper(const std::string& content = "") const;

  std::vector<char> buf_;
  uint32_t offsetInBytes_; // value in bytes
  uint32_t sizeInBytes_; // value in bytes. This is the write head location.
  std::string name_;

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(buf_, offsetInBytes_, sizeInBytes_, name_);
  }
};

template <typename T>
int nElementsToBytes(int nElements) {
  return nElements * sizeof(T);
}

template <>
inline int nElementsToBytes<void>(int nElements) {
  return nElements;
}

template <>
inline int nElementsToBytes<const void>(int nElements) {
  return nElements;
}

// Utility to get buffer address using pointer arithmetic of sizeof(T) for
// adding offset and index.
template <typename T>
T IOBufferAddress(char* buf, int offsetInBytes, int indexInBytes = 0) {
  return reinterpret_cast<T>(buf + offsetInBytes + indexInBytes);
}

// Same as IOBufferAddress() above but for const.
template <typename T>
T IOBufferAddress(const char* buf, int offsetInBytes, int indexInBytes = 0) {
  return reinterpret_cast<T>(buf + offsetInBytes + indexInBytes);
}

template <typename T>
T* IOBuffer::data() {
  return IOBufferAddress<T*>(buf_.data(), offsetInBytes_);
}

template <typename T>
const T* IOBuffer::data() const {
  return IOBufferAddress<const T*>(buf_.data(), offsetInBytes_);
}

template <typename T>
T* IOBuffer::tail() {
  return IOBufferAddress<T*>(buf_.data(), offsetInBytes_, sizeInBytes_);
}

template <typename T>
const T* IOBuffer::tail() const {
  return IOBufferAddress<const T*>(buf_.data(), offsetInBytes_, sizeInBytes_);
}

template <typename T>
void IOBuffer::consume(int size) {
  if (size < 0) {
    std::stringstream ss;
    ss << "Invalid size requested at IOBuffer::consume[](size=" << size << ")";
    throw std::invalid_argument(ss.str());
  }
  const int bytes = size * sizeof(T);
  if (bytes > sizeInBytes_) {
    std::stringstream ss;
    ss << "Request to IOBuffer::consume[](size=" << size
       << " sizeof(T)=" << sizeof(T)
       << ") is greater than internally filled buffer size in bytes="
       << sizeInBytes_;
    throw std::invalid_argument(ss.str());
  }
  offsetInBytes_ += bytes;
  sizeInBytes_ -= bytes;
}

template <typename T>
void IOBuffer::ensure(int size) {
  if (size < 0) {
    std::stringstream ss;
    ss << "Invalid size at IOBuffer::ensure[](size=" << size << ")";
    throw std::invalid_argument(ss.str());
  }

  const int bytes = size * sizeof(T);

  if (tailRoom() >= bytes) {
    return;
  } else if (headRoom() + tailRoom() >= bytes) {
    reset();
  } else {
    reset();
    // increment similar to std::vector
    int finalSize = bytes + buf_.size();
    finalSize = pow(2, ceil(log(finalSize) / log(2)));
    buf_.resize(finalSize);
  }
}

template <typename T>
void IOBuffer::move(int size) {
  if (size < 0) {
    std::stringstream ss;
    ss << "Invalid size at IOBuffer::move[](size=" << size << ")=";
    throw std::invalid_argument(ss.str());
  }
  if (sizeInBytes_ + size * sizeof(T) > buf_.size()) {
    std::stringstream ss;
    ss << "Cannot move beyond end of buffer IOBuffer::move[](size=" << size
       << "): sizeInBytes_=" << sizeInBytes_ << " buf_.size()=" << buf_.size();
    throw std::invalid_argument(ss.str());
  }
  sizeInBytes_ += (size * sizeof(T));
}

template <typename T>
void IOBuffer::write(const T* buf, int size) {
  if (size < 0) {
    std::stringstream ss;
    ss << "Invalid size at IOBuffer::write[](size=" << size << ")=";
    throw std::invalid_argument(ss.str());
  }
  ensure<T>(size);
  std::copy_n(buf, size, tail<T>());
  move<T>(size);
}

template <typename T>
void IOBuffer::writeZero(int size) {
  ensure<T>(size);
  std::fill_n(tail<T>(), size, 0);
  move<T>(size);
}

template <typename T>
int IOBuffer::size() const {
  assert(sizeInBytes_ >= 0);
  return sizeInBytes_ / sizeof(T);
}

template <typename T>
std::string IOBuffer::debugStringWithContent() const {
  std::stringstream ss;
  const T* ptr = data<T>();
  for (int i = 0; i < size<T>(); ++i) {
    ss << ptr[i] << ", ";
  }
  return debugStringHelper(ss.str());
}

} // namespace streaming
} // namespace w2l
