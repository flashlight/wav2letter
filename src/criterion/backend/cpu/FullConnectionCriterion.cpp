/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/FullConnectionCriterion.h"

#include "common/FlashlightUtils.h"
#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cpu/FullConnectionCriterion.h"

using fl::Variable;
using FCC = w2l::cpu::FullConnectionCriterion<float>;

namespace {
// By passing shared_ptr<Context> we avoid copies from forward to backward.
struct Context {
  std::vector<float> transVec;
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
    const std::shared_ptr<Context>& ctx) {
  if (gradVar.type() != f32) {
    throw std::invalid_argument("FCC: grad must be float32");
  }

  auto gradVec = afToVector<float>(gradVar);
  std::vector<float> inputGradVec(B * T * N);
  std::vector<float> transGradVec(N * N);

  FCC::backward(
      B,
      T,
      N,
      ctx->transVec.data(),
      gradVec.data(),
      inputGradVec.data(),
      transGradVec.data(),
      ctx->workspaceVec.data());

  af::array inputGrad(N, T, B, inputGradVec.data());
  af::array transGrad(N, N, transGradVec.data());

  inputs[0].addGrad(Variable(inputGrad, false));
  inputs[1].addGrad(Variable(transGrad, false));
}

Variable FullConnectionCriterion::forward(
    const Variable& inputVar,
    const Variable& targetVar) {
  const auto& transVar = param(0);
  int B = inputVar.dims(2);
  int T = inputVar.dims(1);
  int N = inputVar.dims(0);

  if (N != transVar.dims(0)) {
    throw std::invalid_argument("FCC: input dim doesn't match N");
  } else if (inputVar.type() != f32) {
    throw std::invalid_argument("FCC: input must be float32");
  } else if (targetVar.type() != s32) {
    throw std::invalid_argument("FCC: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(targetVar.array(), T);
  auto ctx = std::make_shared<Context>();
  auto inputVec = afToVector<float>(inputVar);
  auto targetVec = afToVector<int>(targetVar);
  auto targetSizeVec = afToVector<int>(targetSize);
  ctx->transVec = afToVector<float>(transVar);
  std::vector<float> lossVec(B);
  ctx->workspaceVec.assign(FCC::getWorkspaceSize(B, T, N), 0);

  FCC::forward(
      B,
      T,
      N,
      scaleMode_,
      inputVec.data(),
      targetSizeVec.data(),
      ctx->transVec.data(),
      lossVec.data(),
      ctx->workspaceVec.data());

  return Variable(
      af::array(B, lossVec.data()),
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, ctx);
      });
}

} // namespace w2l
