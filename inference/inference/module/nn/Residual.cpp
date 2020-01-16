/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Residual.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Residual::Residual(std::shared_ptr<InferenceModule> module, DataType dataType)
    : module_(module),
      dataType_(dataType),
      identity_(std::make_shared<Identity>()) {
  if (!module) {
    throw std::invalid_argument(
        "Residual::Residual() is called with null module.");
  }
}

Residual::Residual()
    : module_(nullptr),
      dataType_(DataType::UNINITIALIZED),
      identity_(nullptr) {}

std::shared_ptr<ModuleProcessingState> Residual::start(
    std::shared_ptr<ModuleProcessingState> input) {
  // add one more buffer to store a copy of input
  input->buffers().push_back(std::make_shared<IOBuffer>());
  input->buffers().back()->write<char>(
      input->buffer(0)->data<char>(), input->buffer(0)->size<char>());

  std::shared_ptr<ModuleProcessingState> inputCopy = identity_->start(input);

  std::shared_ptr<ModuleProcessingState> output = module_->start(inputCopy);
  assert(output);
  std::shared_ptr<ModuleProcessingState> residualSum = output->next(true, 1);
  assert(residualSum);
  sum(input->buffers().back(), output->buffer(0), residualSum->buffer(0));
  return identity_->start(residualSum);
}

std::shared_ptr<ModuleProcessingState> Residual::run(
    std::shared_ptr<ModuleProcessingState> input) {
  input->buffers().back()->write<char>(
      input->buffer(0)->data<char>(), input->buffer(0)->size<char>());

  std::shared_ptr<ModuleProcessingState> inputCopy = identity_->run(input);

  std::shared_ptr<ModuleProcessingState> output = module_->run(inputCopy);
  assert(output);
  std::shared_ptr<ModuleProcessingState> residualSum = output->next();
  assert(residualSum);
  sum(input->buffers().back(), output->buffer(0), residualSum->buffer(0));
  return identity_->run(residualSum);
}

std::shared_ptr<ModuleProcessingState> Residual::finish(
    std::shared_ptr<ModuleProcessingState> input) {
  input->buffers().back()->write<char>(
      input->buffer(0)->data<char>(), input->buffer(0)->size<char>());

  std::shared_ptr<ModuleProcessingState> inputCopy = identity_->finish(input);

  std::shared_ptr<ModuleProcessingState> output = module_->finish(inputCopy);
  assert(output);
  std::shared_ptr<ModuleProcessingState> residualSum = output->next();
  assert(residualSum);
  sum(input->buffers().back(), output->buffer(0), residualSum->buffer(0));
  return identity_->finish(residualSum);
}

void Residual::setMemoryManager(std::shared_ptr<MemoryManager> memoryManager) {
  InferenceModule::setMemoryManager(memoryManager);
  module_->setMemoryManager(memoryManager);
}

std::string Residual::debugString() const {
  std::stringstream ss;
  ss << "Residual: { ";
  ss << module_->debugString();
  ss << "}";
  return ss.str();
}

void Residual::sum(
    std::shared_ptr<IOBuffer> bufA,
    std::shared_ptr<IOBuffer> bufB,
    std::shared_ptr<IOBuffer> bufC) const {
  switch (dataType_) {
    case DataType::FLOAT: {
      float* aPtr = bufA->data<float>();
      float* bPtr = bufB->data<float>();

      auto len = std::min(bufA->size<float>(), bufB->size<float>());
      bufC->ensure<float>(len);
      float* cPtr = bufC->tail<float>();
      for (int i = 0; i < len; ++i) {
        cPtr[i] = aPtr[i] + bPtr[i];
      }
      bufA->consume<float>(len);
      bufB->consume<float>(len);
      bufC->move<float>(len);
    } break;
    default:
      std::stringstream ss;
      ss << "ReLU for dataType=" << dataTypeString(dataType_)
         << " is not implemented at Relu::run()";
      throw std::runtime_error(ss.str());
  }
}

} // namespace streaming
} // namespace w2l
