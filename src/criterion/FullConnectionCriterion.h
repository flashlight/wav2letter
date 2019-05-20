/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>
#include "CriterionUtils.h"
#include "Defines.h"

namespace w2l {

class FullConnectionCriterion : public fl::BinaryModule {
 public:
  explicit FullConnectionCriterion(
      int N,
      w2l::CriterionScaleMode scalemode = w2l::CriterionScaleMode::NONE);

  fl::Variable forward(const fl::Variable& input, const fl::Variable& target)
      override;

  std::string prettyString() const override;

 private:
  friend class AutoSegmentationCriterion;
  FullConnectionCriterion() = default;

  int N_;
  w2l::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(
      fl::BinaryModule,
      fl::serializeAs<int64_t>(N_),
      scaleMode_)
};

typedef FullConnectionCriterion FCCLoss;

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::FullConnectionCriterion)
