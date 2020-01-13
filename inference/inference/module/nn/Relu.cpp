/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Relu.h"

#include <cassert>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Relu::Relu(DataType dataType) : dataType_(dataType) {
  if (!dataTypeIsValid(dataType)) {
    std::stringstream ss;
    ss << "Invalid dataType at Relu::Relu(dataType(int value)="
       << static_cast<int>(dataType) << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::shared_ptr<ModuleProcessingState> Relu::start(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  assert(input->buffers().size() == 1);
  return input;
}

std::shared_ptr<ModuleProcessingState> Relu::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  assert(input->buffers().size() == 1);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  int nElements = inputBuf->size<char>() / dataTypeNumberOfBytes(dataType_);
  if (nElements == 0) {
    return input;
  }

  switch (dataType_) {
    case DataType::FLOAT: {
      float* inputPtr = inputBuf->data<float>();
      for (int i = 0; i < inputBuf->size<float>(); ++i) {
        inputPtr[i] = std::fmax(inputPtr[i], 0.0);
      }
    } break;
    default:
      std::stringstream ss;
      ss << "ReLU for dataType=" << dataTypeString(dataType_)
         << " is not implemented at Relu::run()";
      throw std::runtime_error(ss.str());
  }

  return input;
}

std::string Relu::debugString() const {
  std::stringstream ss;
  ss << "Relu:{dataType_=" << dataTypeString(dataType_) << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
