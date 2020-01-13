/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "inference/module/ModuleParameter.h"

namespace w2l {
namespace streaming {

ModuleParameter::ModuleParameter(
    DataType type,
    const void* buffer,
    int nElements)
    : type_(type), buffer_(buffer, nElements * dataTypeNumberOfBytes(type)) {
  if (nElements < 0) {
    std::stringstream ss;
    ss << "Invalid nElements at ModuleParameter::ModuleParameter(type="
       << dataTypeString(type) << " nElements=" << nElements << ")";
    throw std::invalid_argument(ss.str());
  }
}

ModuleParameter::ModuleParameter()
    : ModuleParameter(DataType::UNINITIALIZED, nullptr, 0) {}

std::string ModuleParameter::debugString() const {
  std::stringstream ss;
  ss << "ModuleParameter:{type_=" << dataTypeString(type_)
     << " buffer_=" << buffer_.debugString() << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
