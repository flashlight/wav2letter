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

// The createLinear() functions returns a decedent of Linear
// which uses backend accelaration determine by compile time configuration.

class Linear;

// Input matrixes are in col-major format.
std::shared_ptr<Linear> createLinear(
    int nInput,
    int nOutput,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias);

class Linear : public InferenceModule {
 public:
  Linear(int nInput, int nOutput);

  virtual ~Linear() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;
  virtual std::string debugStringWithContent() const;

 protected:
  uint32_t nInput_;
  uint32_t nOutput_;

 private:
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this), nInput_, nOutput_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Linear);
