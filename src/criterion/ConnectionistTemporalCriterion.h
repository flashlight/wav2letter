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

namespace fl {

class ConnectionistTemporalCriterion : public SequenceCriterion {
 public:
  ConnectionistTemporalCriterion(
      w2l::CriterionScaleMode scalemode = w2l::CriterionScaleMode::NONE);

  Variable forward(const Variable& input, const Variable& target) override;

  af::array viterbiPath(const af::array& input) override;

  virtual std::string prettyString() const override;

 private:
  w2l::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, scaleMode_)

  void validate(const Variable& input, const Variable& target);
};

typedef ConnectionistTemporalCriterion CTCLoss;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::ConnectionistTemporalCriterion)
