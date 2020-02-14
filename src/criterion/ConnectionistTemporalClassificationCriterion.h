/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "CriterionUtils.h"
#include "Defines.h"
#include "SequenceCriterion.h"

namespace w2l {

class ConnectionistTemporalClassificationCriterion : public SequenceCriterion {
 public:
  ConnectionistTemporalClassificationCriterion(
      w2l::CriterionScaleMode scalemode = w2l::CriterionScaleMode::NONE);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(const af::array& input) override;
  af::array viterbiPath(const af::array& input, const af::array& target)
      override;

  std::string prettyString() const override;

 private:
  w2l::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, scaleMode_)

  void validate(const fl::Variable& input, const fl::Variable& target);
};

typedef ConnectionistTemporalClassificationCriterion CTCLoss;

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::ConnectionistTemporalClassificationCriterion)
