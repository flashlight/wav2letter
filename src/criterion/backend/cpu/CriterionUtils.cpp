/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/CriterionUtils.h"

#include "common/FlashlightUtils.h"
#include "libraries/criterion/cpu/CriterionUtils.h"
#include "libraries/criterion/cpu/ViterbiPath.h"

using CriterionUtils = w2l::cpu::CriterionUtils<float>;
using ViterbiPath = w2l::cpu::ViterbiPath<float>;

namespace w2l {

af::array viterbiPath(const af::array& input, const af::array& trans) {
  auto B = input.dims(2);
  auto T = input.dims(1);
  auto N = input.dims(0);

  if (N != trans.dims(0) || N != trans.dims(1)) {
    throw std::invalid_argument("viterbiPath: mismatched dims");
  } else if (input.type() != f32) {
    throw std::invalid_argument("viterbiPath: input must be float32");
  } else if (trans.type() != f32) {
    throw std::invalid_argument("viterbiPath: trans must be float32");
  }

  auto inputVec = afToVector<float>(input);
  auto transVec = afToVector<float>(trans);
  std::vector<int> pathVec(B * T);
  std::vector<uint8_t> workspaceVec(ViterbiPath::getWorkspaceSize(B, T, N));

  ViterbiPath::compute(
      B,
      T,
      N,
      inputVec.data(),
      transVec.data(),
      pathVec.data(),
      workspaceVec.data());

  return af::array(T, B, pathVec.data());
}

af::array getTargetSizeArray(const af::array& target, int maxSize) {
  int B = target.dims(1);
  int L = target.dims(0);

  auto targetVec = afToVector<int>(target);
  std::vector<int> targetSizeVec(B);

  CriterionUtils::batchTargetSize(
      B, L, maxSize, targetVec.data(), targetSizeVec.data());

  return af::array(B, targetSizeVec.data());
}

} // namespace w2l
