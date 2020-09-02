/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/module/LMWrapper.h"

using namespace fl;

namespace w2l {

LMWrapper::LMWrapper(
    std::shared_ptr<Module> network,
    const std::vector<int>& dictIndexMap,
    int startIndex)
    : dictIndexMap_(dictIndexMap.size(), dictIndexMap.data()),
      startIndex_(startIndex) {
  add(network);
}

std::vector<Variable> LMWrapper::forward(const std::vector<Variable>& inputs) {
  if (inputs.size() > 2) {
    throw std::invalid_argument("Invalid inputs size");
  }

  // inputs[0] is of size [targetlen, batchsize]
  int U = inputs[0].dims(0);
  int B = inputs[0].dims(1);
  if (U == 0) {
    throw std::invalid_argument("Invalid input variable size");
  }

  auto idxs = af::flat(inputs[0].array());
  auto input = moddims(noGrad(dictIndexMap_(idxs)), {U, B});

  // pad start token
  Variable lmInput = constant(startIndex_, {1, B}, s32, false);
  if (U > 1) {
    lmInput = concatenate({lmInput, input(af::seq(0, U - 2), af::span)}, 0);
  }

  auto logProbOutput = lm()->forward({lmInput}).front();

  // [U, B]
  auto losses = categoricalCrossEntropy(logProbOutput, input, ReduceMode::NONE);

  if (inputs.size() == 2) { // mask padding
    auto endIdx = inputs[1].array();
    af::array i1 = af::range(af::dim4(U, B), 0);
    af::array i2 = tile(af::moddims(endIdx, af::dim4(1, B)), af::dim4(U, 1));
    auto mask = (i1 < i2);
    losses = noGrad(mask) * losses;
  }

  losses = flat(sum(losses, {0}));
  return {losses, logProbOutput};
}

std::string LMWrapper::prettyString() const {
  return "LM: " + lm()->prettyString();
}

} // namespace w2l
