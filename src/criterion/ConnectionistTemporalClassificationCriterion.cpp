/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ConnectionistTemporalClassificationCriterion.h"

using namespace fl;

namespace w2l {

ConnectionistTemporalClassificationCriterion::
    ConnectionistTemporalClassificationCriterion(
        w2l::CriterionScaleMode scalemode /* = w2l::CriterionScaleMode::NONE */)
    : scaleMode_(scalemode) {}

af::array ConnectionistTemporalClassificationCriterion::viterbiPath(
    const af::array& input) {
  af::array bestpath, maxvalues;
  af::max(maxvalues, bestpath, input, 0);
  return af::moddims(bestpath, bestpath.dims(1), bestpath.dims(2));
}

std::string ConnectionistTemporalClassificationCriterion::prettyString() const {
  return "ConnectionistTemporalClassificationCriterion";
}

void ConnectionistTemporalClassificationCriterion::validate(
    const Variable& input,
    const Variable& target) {
  if (input.isempty()) {
    throw(af::exception("CTC: Input cannot be empty"));
  }
  if (target.numdims() > 2) {
    throw(af::exception(
        "CTC: Incorrect dimensions for target. Expected dim4(L, B)"));
  }
  if (input.numdims() > 3) {
    throw(af::exception(
        "CTC: Incorrect dimensions for input. Expected dim4(N, T, B)"));
  }
  if (input.dims(2) != target.dims(1)) {
    throw(af::exception("CTC: Batchsize mismatch for input and target"));
  }
}

} // namespace w2l
