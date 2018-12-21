/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AutoSegmentationCriterion.h"
#include "CriterionUtils.h"

namespace fl {

class LinearSegmentationCriterion : public AutoSegmentationCriterion {
 public:
  explicit LinearSegmentationCriterion(
      intl N,
      w2l::CriterionScaleMode scaleMode = w2l::CriterionScaleMode::NONE)
      : AutoSegmentationCriterion(N, scaleMode) {}

  Variable forward(const Variable& input, const Variable& target) override {
    return AutoSegmentationCriterion::forward(
        input, w2l::getLinearTarget(target, input.dims(1)));
  }

  std::string prettyString() const override {
    return "LinearSegmentationCriterion";
  }

 private:
  LinearSegmentationCriterion() = default;

  FL_SAVE_LOAD_WITH_BASE(AutoSegmentationCriterion)
};

using LinSegCriterion = LinearSegmentationCriterion;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::LinearSegmentationCriterion)
