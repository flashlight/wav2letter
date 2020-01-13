/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstdio>
#include <memory>

#include "inference/common/DataType.h"
#include "inference/common/IOBuffer.h"
#include "inference/module/InferenceModule.h"

namespace w2l {
namespace streaming {

class Relu : public InferenceModule {
 public:
  explicit Relu(DataType dataType);

  virtual ~Relu() override = default;

  // Return the input.
  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  // Working directly on the input.
  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

 protected:
  DataType dataType_;

 private:
  friend class cereal::access;

  // Required for Cereal serialization.
  Relu() : Relu(DataType::UNINITIALIZED) {}

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this), dataType_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Relu);
