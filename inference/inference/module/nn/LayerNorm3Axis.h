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

// Copy of LayerNorm with possibility to normalize over 3 axis
// Implemented as separate class to not to break serialization.
class LayerNorm3Axis : public InferenceModule {
 public:
  LayerNorm3Axis(int32_t featureSize, float alpha, float beta, int32_t axis = 2);

  virtual ~LayerNorm3Axis() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

 protected:
  int32_t featureSize_;
  int32_t axis_;
  float alpha_;
  float beta_;

 private:
  friend class cereal::access;

  LayerNorm3Axis();

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this), featureSize_, alpha_, beta_, axis_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::LayerNorm3Axis);
