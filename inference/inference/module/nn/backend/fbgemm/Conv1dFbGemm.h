/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <fbgemm/FbgemmFP16.h>
#include <memory>
#include <string>

#include "inference/module/ModuleParameter.h"
#include "inference/module/ModuleProcessingState.h"
#include "inference/module/nn/Conv1d.h"
#include "inference/module/nn/backend/fbgemm/PackedGemmMatrixFP16.h"

namespace w2l {
namespace streaming {

class Conv1dFbGemm : public Conv1d {
 public:
  // weights is freed after we read its content into internal float 16 packed
  // representation.
  Conv1dFbGemm(
      int inChannels,
      int outChannels,
      int kernelSize,
      int stride,
      int rightPadding,
      int leftPadding,
      int groups,
      std::shared_ptr<ModuleParameter> weights,
      std::shared_ptr<ModuleParameter> bias);

  virtual ~Conv1dFbGemm() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> finish(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

 protected:
  void init(std::shared_ptr<ModuleParameter> weights);

  std::shared_ptr<ModuleParameter> bias_;
  std::shared_ptr<fbgemm::PackedGemmMatrixFP16> packedWeights_;

 private:
  friend class cereal::access;

  Conv1dFbGemm(); // Used by Cereal for serialization.

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<Conv1d>(this), bias_, packedWeights_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Conv1dFbGemm);
