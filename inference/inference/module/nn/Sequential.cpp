/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Sequential.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Sequential::Sequential(std::vector<std::shared_ptr<InferenceModule>> modules)
    : modules_(modules) {}

Sequential::Sequential() {}

void Sequential::add(std::shared_ptr<InferenceModule> module) {
  modules_.push_back(module);
}

std::shared_ptr<ModuleProcessingState> Sequential::start(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->start(intermediateInput);
  }
  return intermediateInput;
}

std::shared_ptr<ModuleProcessingState> Sequential::run(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->run(intermediateInput);
  }
  return intermediateInput;
}

std::shared_ptr<ModuleProcessingState> Sequential::finish(
    std::shared_ptr<ModuleProcessingState> input) {
  std::shared_ptr<ModuleProcessingState> intermediateInput = input;
  for (auto& module : modules_) {
    assert(module);
    intermediateInput = module->finish(intermediateInput);
  }
  return intermediateInput;
}

void Sequential::setMemoryManager(
    std::shared_ptr<MemoryManager> memoryManager) {
  InferenceModule::setMemoryManager(memoryManager);
  for (auto module : modules_) {
    module->setMemoryManager(memoryManager);
  }
}

std::string Sequential::debugString() const {
  std::stringstream ss;
  ss << "Sequential: { \n";
  for (auto& module : modules_) {
    ss << module->debugString() << "\n";
  }

  ss << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
