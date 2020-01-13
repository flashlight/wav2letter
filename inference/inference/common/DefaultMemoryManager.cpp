/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/common/DefaultMemoryManager.h"

#include <cstdlib>

namespace w2l {
namespace streaming {

DefaultMemoryManager::DefaultMemoryManager() {}

void* DefaultMemoryManager::allocate(size_t sizeInBytes) {
  return std::malloc(sizeInBytes);
}

void DefaultMemoryManager::free(void* ptr) {
  std::free(ptr);
}

std::string DefaultMemoryManager::debugString() const {
  return "Default Memory Manager";
}

} // namespace streaming
} // namespace w2l
