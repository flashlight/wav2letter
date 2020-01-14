/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <sstream>
#include <stdexcept>

#include "inference/common/DataType.h"

namespace w2l {
namespace streaming {

const std::string dataTypeString(DataType type) {
  static const cpp::enum_unordered_map<DataType, const char*> typeToName = {
      {DataType::UNINITIALIZED, "UNINITIALIZED"},
      {DataType::FLOAT, "FLOAT"},
      {DataType::FLOAT16, "FLOAT16"}};
  auto itr = typeToName.find(type);
  if (itr == typeToName.end()) {
    std::ostringstream ss;
    ss << "Invalid type at dataTypeString(type=" << static_cast<int>(type)
       << ")";
    throw std::out_of_range(ss.str());
  }
  return itr->second;
}

int dataTypeNumberOfBytes(DataType type) {
  static const cpp::enum_unordered_map<DataType, int> typeToSize = {
      {DataType::UNINITIALIZED, 0},
      {DataType::FLOAT, 4},
      {DataType::FLOAT16, 2}};
  auto itr = typeToSize.find(type);
  if (itr == typeToSize.end()) {
    std::ostringstream ss;
    ss << "Invalid type at dataTypeString(type=" << static_cast<int>(type)
       << ")";
    throw std::out_of_range(ss.str());
  }
  return itr->second;
}

bool dataTypeIsValid(DataType type) {
  try {
    dataTypeString(type);
    return true;
  } catch (const std::out_of_range&) {
    return false;
  }
}

} // namespace streaming
} // namespace w2l
