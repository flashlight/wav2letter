/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/ForceAlignmentCriterion.h"

namespace w2l {

ForceAlignmentCriterion::ForceAlignmentCriterion(
    int N,
    w2l::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FAC: Size of transition matrix is less than 0");
  }
  auto transition = fl::constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

std::string ForceAlignmentCriterion::prettyString() const {
  return "ForceAlignmentCriterion";
}

} // namespace w2l
