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

#include "inference/common/MemoryManager.h"

namespace w2l {
namespace streaming {

class DefaultMemoryManager : public MemoryManager {
 public:
  DefaultMemoryManager();

  virtual ~DefaultMemoryManager() override = default;

  std::string debugString() const override;

 protected:
  void* allocate(size_t sizeInBytes) override;
  void free(void* ptr) override;
};

} // namespace streaming
} // namespace w2l
