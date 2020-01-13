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

class Sequential : public InferenceModule {
 public:
  Sequential();

  explicit Sequential(std::vector<std::shared_ptr<InferenceModule>> modules);

  virtual ~Sequential() override = default;

  void add(std::shared_ptr<InferenceModule> module);

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> finish(
      std::shared_ptr<ModuleProcessingState> input) override;

  void setMemoryManager(std::shared_ptr<MemoryManager> memoryManager) override;

  virtual std::string debugString() const override;

 protected:
  std::vector<std::shared_ptr<InferenceModule>> modules_;

 private:
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this), modules_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Sequential);
