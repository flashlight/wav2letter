/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/ForceAlignmentCriterion.h"

#include <flashlight/common/cuda.h>

#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cuda/ForceAlignmentCriterion.cuh"

using fl::Variable;
using FAC = w2l::cuda::ForceAlignmentCriterion<float>;

namespace w2l {

static void backward(
    std::vector<Variable>& inputs,
    const Variable& gradVar,
    int B,
    int T,
    int N,
    int L,
    const af::array& target,
    const af::array& targetSize,
    af::array& workspace) {
  if (gradVar.type() != f32) {
    throw std::invalid_argument("FAC: grad must be float32");
  }

  const auto& grad = gradVar.array();
  af::array inputGrad(N, T, B, f32);
  af::array transGrad(N, N, f32);

  {
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr gradRaw(grad);
    fl::DevicePtr inputGradRaw(inputGrad);
    fl::DevicePtr transGradRaw(transGrad);
    fl::DevicePtr workspaceRaw(workspace);
    FAC::backward(
        B,
        T,
        N,
        L,
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(gradRaw.get()),
        static_cast<float*>(inputGradRaw.get()),
        static_cast<float*>(transGradRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }

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

  const auto& input = inputVar.array();
  const auto& target = targetVar.array();
  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar.array();
  af::array loss(B, f32);
  af::array workspace(FAC::getWorkspaceSize(B, T, N, L), u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr lossRaw(loss);
    fl::DevicePtr workspaceRaw(workspace);

    FAC::forward(
        B,
        T,
        N,
        L,
        scaleMode_,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<float*>(lossRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }

  return Variable(
      loss,
      {inputVar.withoutData(), transVar.withoutData()},
      [=](std::vector<Variable>& inputs, const Variable& gradVar) mutable {
        backward(inputs, gradVar, B, T, N, L, target, targetSize, workspace);
      });
}

af::array ForceAlignmentCriterion::viterbiPath(
    const af::array& input,
    const af::array& target) {
  int N = input.dims(0);
  int T = input.dims(1);
  int B = input.dims(2);
  int L = target.dims(0);

  std::vector<std::vector<int>> bestPaths;
  const auto& transVar = param(0);

  if (N != transVar.dims(0)) {
    throw std::invalid_argument("FAC: input dim doesn't match N:");
  } else if (input.type() != f32) {
    throw std::invalid_argument("FAC: input must be float32");
  } else if (target.type() != s32) {
    throw std::invalid_argument("FAC: target must be int32");
  }

  const auto& targetSize = getTargetSizeArray(target, T);
  const auto& trans = transVar.array();
  af::array bestPathsVar(T, B, s32);
  af::array workspace(FAC::getWorkspaceSize(B, T, N, L), u8);

  {
    fl::DevicePtr inputRaw(input);
    fl::DevicePtr targetRaw(target);
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr transRaw(trans);
    fl::DevicePtr bestPathsRaw(bestPathsVar);
    ;
    fl::DevicePtr workspaceRaw(workspace);

    FAC::viterbiPath(
        B,
        T,
        N,
        L,
        static_cast<const float*>(inputRaw.get()),
        static_cast<const int*>(targetRaw.get()),
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<const float*>(transRaw.get()),
        static_cast<int*>(bestPathsRaw.get()),
        workspaceRaw.get(),
        fl::cuda::getActiveStream());
  }
  return bestPathsVar;
}

} // namespace w2l
