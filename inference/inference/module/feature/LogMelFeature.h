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
#include "libraries/feature/FeatureParams.h"
#include "libraries/feature/Mfsc.h"

namespace w2l {
namespace streaming {

class LogMelFeature : public InferenceModule {
 public:
  explicit LogMelFeature(
      int numFilters,
      int frameSizeMs = 25,
      int frameShiftMs = 10,
      int samplingFreq = 16000);

  virtual ~LogMelFeature() override = default;

  std::shared_ptr<ModuleProcessingState> start(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::shared_ptr<ModuleProcessingState> run(
      std::shared_ptr<ModuleProcessingState> input) override;

  std::string debugString() const override;

 private:
  int32_t numFilters_;
  int32_t frameSizeMs_;
  int32_t frameShiftMs_;
  int32_t samplingFreq_;
  FeatureParams featParams_;
  std::shared_ptr<Mfsc> mfscFeaturizer_;

  LogMelFeature();

  void init();

  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this),
       numFilters_,
       frameSizeMs_,
       frameShiftMs_,
       samplingFreq_);
    if (!mfscFeaturizer_) {
      init();
    }
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::LogMelFeature);
