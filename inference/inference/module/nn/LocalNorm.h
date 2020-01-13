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

#include "inference/common/IOBuffer.h"
#include "inference/module/InferenceModule.h"
#include "inference/module/ModuleParameter.h"

namespace w2l {
namespace streaming {

class LocalNorm : public InferenceModule {
 public:
  LocalNorm(int featureSize, int leftContextSize, int rightContextSize);

  virtual ~LocalNorm() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

 protected:
  int32_t featureSize_;
  int32_t leftContextSize_;
  int32_t rightContextSize_; // Right context not supported currently

 private:
  friend class cereal::access;

  LocalNorm();

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this),
       featureSize_,
       leftContextSize_,
       rightContextSize_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::LocalNorm);
