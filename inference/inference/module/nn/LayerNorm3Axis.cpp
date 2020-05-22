/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/LayerNorm3Axis.h"

#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include "inference/common/Functions.h"

namespace {
constexpr const float kEpsilon = 1e-5; // min stddev to avoid division by zero
}

namespace w2l {
namespace streaming {

LayerNorm3Axis::LayerNorm3Axis() : LayerNorm3Axis(1, 1.0, 1.0, 2) {}

LayerNorm3Axis::LayerNorm3Axis(int32_t featureSize, float alpha, float beta, int32_t axis)
    : featureSize_(featureSize), axis_(axis), alpha_(alpha), beta_(beta) {
  if (featureSize <= 0 || (axis != 2 && axis != 3)) {
    std::stringstream ss;
    ss << "Invalid argument at LayerNorm3Axis::LayerNorm3Axis(featureSize=" << featureSize
       << " alpha=" << alpha << " beta_=" << beta << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::shared_ptr<ModuleProcessingState> LayerNorm3Axis::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> LayerNorm3Axis::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);

  if (inputBuf->size<float>()==0)
    return output;

  // in case of 3 axis normalization we consider all data as features
  // in case of 2 axis normalization we consider features for each time step separately
  auto featureSize = axis_==3 ? inputBuf->size<float>() : featureSize_;
  const int T = inputBuf->size<float>() / featureSize;

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  const int N = T * featureSize;
  outputBuf->ensure<float>(N);
  float mean = 0.0, stddev = 0.0;
  for (int t = 0; t < T; ++t) {
    const float* inPtr = inputBuf->data<float>();
    float* outPtr = outputBuf->tail<float>();
    meanAndStddev(inPtr, featureSize, /* output */ mean, /* output */ stddev);

    if (stddev <= kEpsilon) {
      stddev = 1.0;
    }
    meanNormalize(
        inPtr, featureSize, mean, stddev, alpha_, beta_, /* output */ outPtr);

    inputBuf->consume<float>(featureSize);
    outputBuf->move<float>(featureSize);
  }
  outputBuf->dim = inputBuf->dim;
  return output;
}

std::string LayerNorm3Axis::debugString() const {
  std::stringstream ss;
  ss << "LayerNorm3Axis:{featureSize=" << featureSize_ << " alpha=" << alpha_
     << " beta_=" << beta_ << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
