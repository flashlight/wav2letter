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
#include "inference/module/nn/Identity.h"

namespace w2l {
namespace streaming {

class Residual : public InferenceModule {
 public:
  Residual(std::shared_ptr<InferenceModule> module, DataType dataType);

  virtual ~Residual() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> finish(
      std::shared_ptr<ModuleProcessingState> input) override;

  void setMemoryManager(std::shared_ptr<MemoryManager> memoryManager) override;

  std::string debugString() const override;

 protected:
  std::shared_ptr<InferenceModule> module_;
  DataType dataType_;
  std::shared_ptr<Identity> identity_;

 private:
  friend class cereal::access;

  Residual(); // Used by Cereal for serialization.

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this),
       module_,
       dataType_,
       identity_);
  }

  // bufC = bufA + bufB
  void sum(
      std::shared_ptr<IOBuffer> bufA,
      std::shared_ptr<IOBuffer> bufB,
      std::shared_ptr<IOBuffer> bufC) const;
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Residual);
