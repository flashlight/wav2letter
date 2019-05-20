/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/ForceAlignmentCriterion.h"

#include "common/Utils.h"
#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cpu/ForceAlignmentCriterion.h"

using fl::Variable;
using FAC = w2l::cpu::ForceAlignmentCriterion<float>;

namespace {
// By passing shared_ptr<Context> we avoid copies from forward to backward.
struct Context {
  std::vector<int> targetVec;
  std::vector<int> targetSizeVec;
  std::vector<uint8_t> workspaceVec;
};
} // namespace

namespace w2l {

static void backward(
    std::vector<Variable>& inputs,
    const Variable& gradVar,
    int B,
    int T,
    int N,
    int L,
    const std::shared_ptr<Context>& ctx) {
  if (gradVar.type() != f32) {
    throw std::invalid_argument("FAC: grad must be float32");
  }

  auto gradVec = afToVector<float>(gradVar);
  std::vector<float> inputGradVec(B * T * N);
  std::vector<float> transGradVec(N * N);

  FAC::backward(
      B,
      T,
      N,
      L,
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      gradVec.data(),
      inputGradVec.data(),
      transGradVec.data(),
      ctx->workspaceVec.data());

  af::array inputGrad(N, T, B, inputGradVec.data());
  af::array transGrad(N, N, transGradVec.data());

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable ForceAlignmentCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dims(2);
  int T = inputVar.dims(1);
  int N = inputVar.dims(0);
  int L = targetVar.dims(0);

  if (N != transVar.dims(0)) {
    throw std::invalid_argument("FAC: input dim doesn't match N");
  } else if (inputVar.type() != f32) {
    throw std::invalid_argument("FAC: input must be float32");
  } else if (targetVar.type() != s32) {
    throw std::invalid_argument("FAC: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(targetVar.array(), T);
  auto ctx = std::make_shared<Context>();
  auto inputVec = afToVector<float>(inputVar);
  ctx->targetVec = afToVector<int>(targetVar);
  ctx->targetSizeVec = afToVector<int>(targetSize);
  auto transVec = afToVector<float>(transVar);
  std::vector<float> lossVec(B);
  ctx->workspaceVec.assign(FAC::getWorkspaceSize(B, T, N, L), 0);

  FAC::forward(
      B,
      T,
      N,
      L,
      scaleMode_,
      inputVec.data(),
      ctx->targetVec.data(),
      ctx->targetSizeVec.data(),
      transVec.data(),
      lossVec.data(),
      ctx->workspaceVec.data());

  return Variable(
      af::array(B, lossVec.data()),
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) {
        backward(inputs, gradVar, B, T, N, L, ctx);
      });
}

} // namespace w2l
