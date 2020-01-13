/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <memory>

namespace w2l {
namespace streaming {

// Derived classes are a good place to put memory management optimizations,
// policies, stats, publish metrices, ect. Memory can be on the host or the
// device.
class MemoryManager {
 public:
  MemoryManager() {}

  virtual ~MemoryManager() = default;

  virtual std::string debugString() const = 0;

  template <typename T>
  std::shared_ptr<T> makeShared(size_t size) {
    auto deleter = [this](T* ptr) { this->free(ptr); };
    return std::shared_ptr<T>((T*)allocate(size * sizeof(T)), deleter);
  }

 protected:
  virtual void* allocate(size_t sizeInBytes) = 0;
  virtual void free(void* ptr) = 0;
};

} // namespace streaming
} // namespace w2l
