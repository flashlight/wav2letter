/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "inference/common/IOBuffer.h"

namespace w2l {
namespace streaming {

// In streaming inference the input is buffered until we have a minimal
// size for processing. Module input may consist of one more buffers. These
// buffers are part of the modules stream processing state.
class ModuleProcessingState {
 public:
  explicit ModuleProcessingState(int numOfBuffers);

  std::vector<std::shared_ptr<IOBuffer>>& buffers();

  std::shared_ptr<IOBuffer> buffer(int idx);

  // A modules knows the number of output buffers it processes. Before first
  // inference pass the list of ModuleProcessingState has only the input
  // element. When first calling Module::start() on the first module in the
  // processing graph, each module in that graph creates the buffers for its own
  // output, which is also the input for the next module in the graph. It does
  // so by calling next(true, numOfOutputsForCurrentModule).
  std::shared_ptr<ModuleProcessingState> next(
      bool createIfNotExists = false,
      int numOfBuffers = 0);

 private:
  std::vector<std::shared_ptr<IOBuffer>> buffers_;
  std::shared_ptr<ModuleProcessingState> next_;
};

} // namespace streaming
} // namespace w2l
