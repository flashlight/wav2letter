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
#include "ForceAlignmentCriterion.h"
#include "FullConnectionCriterion.h"
#include "SequenceCriterion.h"

namespace fl {

class AutoSegmentationCriterion : public SequenceCriterion {
 public:
  explicit AutoSegmentationCriterion(
      intl N,
      w2l::CriterionScaleMode scalemode = w2l::CriterionScaleMode::NONE,
      double transdiag = 0.0)
      : N_(N),
        scaleMode_(scalemode),
        fac_(ForceAlignmentCriterion(N, scalemode)),
        fcc_(FullConnectionCriterion(N, scalemode)) {
    if (N_ <= 0) {
      throw af::exception("ASG: N is zero or negative.");
    }
    Variable transition(transdiag * af::identity(af::dim4(N_, N_)), true);
    params_ = {transition};
    syncTransitions();
  }

  Variable forward(const Variable& input, const Variable& target) override {
    return fcc_.forward(input, target) - fac_.forward(input, target);
  }

  af::array viterbiPath(const af::array& input) override {
    return w2l::viterbiPath(input, params_[0].array());
  }

  void setParams(const Variable& var, int position) override {
    Module::setParams(var, position);
    syncTransitions();
  }

  std::string prettyString() const override {
    return "AutoSegmentationCriterion";
  }

 protected:
  AutoSegmentationCriterion() = default;

  void syncTransitions() {
    fac_.setParams(params_[0], 0);
    fcc_.setParams(params_[0], 0);
  }

 private:
  intl N_;
  w2l::CriterionScaleMode scaleMode_;
  ForceAlignmentCriterion fac_;
  FullConnectionCriterion fcc_;

  FL_SAVE_LOAD_WITH_BASE(SequenceCriterion, N_, scaleMode_, fac_, fcc_)
};

using ASGLoss = AutoSegmentationCriterion;

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AutoSegmentationCriterion)
