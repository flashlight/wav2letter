/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/ModuleProcessingState.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

ModuleProcessingState::ModuleProcessingState(int numOfBuffers) {
  // Typically a module has a single input and a single output. While a module
  // may have more, it is unlikely to have more than a few.
  if (numOfBuffers <= 0 || numOfBuffers > 100) {
    std::stringstream ss;
    ss << "Invalid numOfBuffers at ModuleProcessingState::ModuleProcessingState(numOfBuffers="
       << numOfBuffers << ")";
    throw std::invalid_argument(ss.str());
  }
  while (buffers_.size() < numOfBuffers) {
    buffers_.push_back(std::make_shared<IOBuffer>());
  }
}

std::vector<std::shared_ptr<IOBuffer>>& ModuleProcessingState::buffers() {
  return buffers_;
}

std::shared_ptr<IOBuffer> ModuleProcessingState::buffer(int idx) {
  if (idx < 0 || idx >= buffers_.size()) {
    std::stringstream ss;
    ss << "Invalid index " << idx
       << " for ModuleProcessingState with buffer size " << buffers_.size();
    throw std::invalid_argument(ss.str());
  }
  return buffers_[idx];
}

std::shared_ptr<ModuleProcessingState> ModuleProcessingState::next(
    bool createIfNotExists,
    int numOfBuffers) {
  if (!next_ && createIfNotExists) {
    next_ = std::make_shared<ModuleProcessingState>(numOfBuffers);
  }
  return next_;
}

} // namespace streaming
} // namespace w2l
