/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace w2l {

/**
 * Partitions a contiguous block of memory into aligned arrays.
 * Can be used for either host or device memory.
 *
 * Usage: first create Workspace(nullptr), request() some arrays, then call
 * requiredSize(). Next, allocate memory of that size. Finally, create
 * Workspace(ptr) and request() the same sequence of arrays.
 */
template <size_t Alignment = 256>
class Workspace {
 public:
  explicit Workspace(void* workspacePtr)
      : workspacePtr_(reinterpret_cast<uintptr_t>(workspacePtr)), offset_(0) {
    align();
  }

  template <class T>
  T* request(size_t s0, size_t s1 = 1, size_t s2 = 1, size_t s3 = 1) {
    align();
    auto p = reinterpret_cast<T*>(workspacePtr_ + offset_);
    offset_ += sizeof(T) * s0 * s1 * s2 * s3;
    return p;
  }

  template <class T>
  void request(T** p, size_t s0, size_t s1 = 1, size_t s2 = 1, size_t s3 = 1) {
    *p = request<T>(s0, s1, s2, s3);
  }

  size_t requiredSize() const {
    // Add extra bytes in case the initial `workspacePtr` isn't aligned
    return offset_ + Alignment - 1;
  }

 private:
  void align() {
    // Pad until `workspacePtr_ + offset_` is a multiple of `Alignment`
    offset_ +=
        Alignment - 1 - (workspacePtr_ + offset_ + Alignment - 1) % Alignment;
  }

  const uintptr_t workspacePtr_;
  size_t offset_;
};

} // namespace w2l
