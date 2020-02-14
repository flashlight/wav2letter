/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include "ConnectionistTemporalClassificationCriterion.h"

#include "common/FlashlightUtils.h"
#include "libraries/criterion/cpu/ConnectionistTemporalClassificationCriterion.h"

using CTC = w2l::cpu::ConnectionistTemporalClassificationCriterion<float>;

namespace {

struct CTCContext {
  std::vector<int> targetVec;
  std::vector<int> targetSizeVec;
  std::vector<uint8_t> workspaceVec;
};

af::array logSoftmax(const af::array& input, const int dim) {
  af::array maxvals = max((input), dim);
  af::dim4 tiledims(1, 1, 1, 1);
  if (dim > 3) {
    throw std::invalid_argument("logSoftmax: Dimension must be less than 3");
  }
  tiledims[dim] = input.dims(dim);
  // Compute log softmax.
  // Subtracting then adding maxvals is for numerical stability.
  auto result = input -
      tile(log(sum(exp(input - tile(maxvals, tiledims)), dim)) + maxvals,
           tiledims);
  result.eval();
  return result;
};

} // namespace

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

af::array ConnectionistTemporalClassificationCriterion::viterbiPath(
    const af::array& inputVar,
    const af::array& targetVar) {
  int N = inputVar.dims(0);
  int T = inputVar.dims(1);
  int B = inputVar.dims(2);
  int L = targetVar.dims(0);

  const af::array targetSize = getTargetSizeArray(targetVar, T);
  std::shared_ptr<CTCContext> ctx = std::make_shared<CTCContext>();
  af::array softmax = logSoftmax(inputVar, 0);
  std::vector<float> inputVec = afToVector<float>(softmax);
  ctx->targetVec = afToVector<int>(targetVar);
  ctx->targetSizeVec = afToVector<int>(targetSize);
  ctx->workspaceVec.assign(CTC::getWorkspaceSize(B, T, N, L), 0);
  std::vector<int> bestPaths(B * T);
  CTC::viterbi(
      B,
      T,
      N,
      L,
      inputVec.data(),
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      bestPaths.data(),
      ctx->workspaceVec.data());
  af::array result(T, B, bestPaths.data());
  return result;
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
