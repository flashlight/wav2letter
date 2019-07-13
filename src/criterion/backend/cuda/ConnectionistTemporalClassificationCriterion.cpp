/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <ctc.h> // warpctc

#include <flashlight/autograd/autograd.h>
#include <flashlight/common/cuda.h>

#include "common/FlashlightUtils.h"
#include "criterion/ConnectionistTemporalClassificationCriterion.h"
#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cuda/CriterionUtils.cuh"

using namespace fl;

using CriterionUtils = w2l::cuda::CriterionUtils<float>;

namespace w2l {

namespace {
inline void throw_on_error(ctcStatus_t status, const char* message) {
  if (status != CTC_STATUS_SUCCESS) {
    throw std::runtime_error(
        message + (", stat = " + std::string(ctcGetStatusString(status))));
  }
}
} // namespace

std::vector<Variable> ConnectionistTemporalClassificationCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];
  validate(input, target);
  const int N = input.dims(0);
  const int T = input.dims(1);
  const int B = input.dims(2);
  const int batchL = target.dims(0);
  auto stream = fl::cuda::getActiveStream();

  ctcOptions options;
  options.loc = CTC_GPU;
  options.stream = stream;
  options.blank_label = N - 1;

  af::array inputarr(N, B, T, input.type());
  inputarr(af::span, af::span, af::span) = af::reorder(input.array(), 0, 2, 1);

  af::array grad;
  if (input.isCalcGrad()) {
    grad = af::constant(0.0, inputarr.dims(), inputarr.type());
  }

  std::vector<int> inputLengths(B, T);
  std::vector<int> labels;
  std::vector<int> labelLengths;
  std::vector<int> batchTargetVec(target.elements());
  target.host(batchTargetVec.data());

  af::array targetSize(B, s32);
  af::array scale(B, f32);

  {
    fl::DevicePtr targetRaw(target.array());
    fl::DevicePtr targetSizeRaw(targetSize);
    fl::DevicePtr scaleRaw(scale);

    CriterionUtils::batchTargetSize(
        B,
        batchL,
        batchL,
        static_cast<const int*>(targetRaw.get()),
        static_cast<int*>(targetSizeRaw.get()),
        stream);

    CriterionUtils::computeScale(
        B,
        T,
        N,
        scaleMode_,
        static_cast<const int*>(targetSizeRaw.get()),
        static_cast<float*>(scaleRaw.get()),
        stream);
  }

  auto batchTargetSizeVec = afToVector<int>(targetSize);
  auto batchScaleVec = afToVector<float>(scale);

  for (int b = 0; b < B; ++b) {
    const int* targetVec = batchTargetVec.data() + b * batchL;
    int L = batchTargetSizeVec[b];

    // A heuristic to modify target length to be able to compute CTC loss
    L = std::min(L, T);
    const int R = w2l::countRepeats(targetVec, L);
    L = std::min(L + R, T) - R;

    labelLengths.push_back(L);
    for (int l = 0; l < L; ++l) {
      labels.push_back(targetVec[l]);
    }
  }
  af::array batchScales(B, batchScaleVec.data());

  size_t workspace_size;
  throw_on_error(
      get_workspace_size(
          labelLengths.data(),
          inputLengths.data(),
          N,
          B,
          options,
          &workspace_size),
      "Error: get_workspace_size");

  af::array workspace(workspace_size, af::dtype::b8);

  std::vector<float> costs(B, 0.0);
  {
    DevicePtr inputarrraw(inputarr);
    DevicePtr gradraw(grad);
    DevicePtr workspaceraw(workspace);
    throw_on_error(
        compute_ctc_loss(
            (float*)inputarrraw.get(),
            (float*)gradraw.get(),
            labels.data(),
            labelLengths.data(),
            inputLengths.data(),
            N,
            B,
            costs.data(),
            workspaceraw.get(),
            options),
        "Error: compute_ctc_loss");
  }

  af::array result(B, costs.data());

  result = result * batchScales;

  auto gradFunc = [grad, batchScales](
                      std::vector<Variable>& moduleInputs,
                      const Variable& grad_output) {
    auto gradScales = grad_output.array() * batchScales;
    auto& in = moduleInputs[0];
    gradScales = af::tile(
        moddims(gradScales, 1, grad_output.dims(0), 1),
        in.dims(0),
        1,
        in.dims(1));
    moduleInputs[0].addGrad(
        Variable(af::reorder(grad * gradScales, 0, 2, 1), false));
  };

  return {Variable(result, {input, target}, gradFunc)};
}

} // namespace w2l
