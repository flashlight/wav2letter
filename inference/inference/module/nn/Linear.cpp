/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Linear.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Linear::Linear(int nInput, int nOutput) : nInput_(nInput), nOutput_(nOutput) {
  if (nInput < 0 || nOutput < 0) {
    std::stringstream ss;
    ss << "Invalid argument at Linear::Linear(nInput=" << nInput
       << " nOutput=" << nOutput << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::shared_ptr<ModuleProcessingState> Linear::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::string Linear::debugString() const {
  std::stringstream ss;
  ss << "Linear:{nInput_=" << nInput_ << " nOutput_=" << nOutput_ << "}";
  return ss.str();
}

std::string Linear::debugStringWithContent() const {
  return debugString();
}

} // namespace streaming
} // namespace w2l
