/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/access.hpp>
#include <cstdio>
#include <string>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"

namespace w2l {
namespace streaming {

class ModuleParameter {
 public:
  // Copy nElements from buffer into buffer_. Element's size if determined by
  // type.
  ModuleParameter(DataType type, const void* buffer, int nElements);

  DataType type_;
  IOBuffer buffer_;

  std::string debugString() const;

 private:
  friend class cereal::access;

  ModuleParameter(); // Used by Cereal for deserialization.

  template <class Archive>
  void serialize(Archive& ar) {
    ar(type_, buffer_);
  }
};

} // namespace streaming
} // namespace w2l
