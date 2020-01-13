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
#include <memory>
#include <utility>

#include "inference/common/IOBuffer.h"
#include "inference/module/InferenceModule.h"
#include "inference/module/ModuleParameter.h"

namespace w2l {
namespace streaming {

// The createConv1d() functions returns a decedent of Conv1d
// which uses backend accelaration determine by compile time configuration.

class Conv1d;

// Input matrixes are in col-major format.
std::shared_ptr<Conv1d> createConv1d(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    int padding,
    int groups,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias);

// This constructor allows for diferent right and left padding.
std::shared_ptr<Conv1d> createConv1d(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    const std::pair<int, int> padding, // {leftPadding, rightPadding}
    int groups,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias);

class Conv1d : public InferenceModule {
 public:
  Conv1d(
      int inChannels,
      int outChannels,
      int kernelSize,
      int stride,
      int rightPadding,
      int leftPadding,
      int groups);

  virtual ~Conv1d() override = default;

  std::string debugString() const override;

 protected:
  uint32_t inChannels_;
  uint32_t outChannels_;
  uint32_t kernelSize_;
  uint32_t stride_;
  uint32_t rightPadding_;
  uint32_t leftPadding_;
  uint32_t groups_;

 private:
  friend class cereal::access;

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<InferenceModule>(this),
       groups_,
       inChannels_,
       outChannels_,
       kernelSize_,
       stride_,
       rightPadding_,
       leftPadding_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::Conv1d);
