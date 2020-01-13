/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/LocalNorm.h"

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

LocalNorm::LocalNorm() : LocalNorm(1, 0, 0) {}

LocalNorm::LocalNorm(
    int32_t featureSize,
    int32_t leftContextSize,
    int32_t rightContextSize)
    : featureSize_(featureSize),
      leftContextSize_(leftContextSize),
      rightContextSize_(rightContextSize) {
  if (featureSize <= 0 || leftContextSize < 0 || rightContextSize != 0) {
    std::stringstream ss;
    ss << "Invalid argument at LocalNorm::LocalNorm("
       << " featureSize=" << featureSize
       << " leftContextSize=" << leftContextSize
       << " rightContextSize=" << rightContextSize << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::shared_ptr<ModuleProcessingState> LocalNorm::start(
    std::shared_ptr<ModuleProcessingState> input) {
  // Create 3 output buffers to store output, sum of input per feature frame,
  // sum of squared input per feature frame respectively
  return input->next(true, 3);
}

std::shared_ptr<ModuleProcessingState> LocalNorm::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);
  const int nFeatFrames = inputBuf->size<float>() / featureSize_;
  if (nFeatFrames == 0) {
    return output;
  }

  assert(output->buffers().size() >= 3);
  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);
  std::shared_ptr<IOBuffer> sumBuf = output->buffer(1);
  std::shared_ptr<IOBuffer> sqSumBuf = output->buffer(2);
  assert(sumBuf && sqSumBuf);

  const int outputSize = nFeatFrames * featureSize_;
  outputBuf->ensure<float>(outputSize);

  for (int t = 0; t < nFeatFrames; ++t) {
    const float* inPtr = inputBuf->data<float>();
    float* outPtr = outputBuf->tail<float>();

    float curSum = std::accumulate(inPtr, inPtr + featureSize_, 0.0);
    float curSqSum =
        std::inner_product(inPtr, inPtr + featureSize_, inPtr, 0.0);
    sumBuf->write(&curSum, 1);
    sqSumBuf->write(&curSqSum, 1);

    int totalSize = sumBuf->size<float>();
    assert(totalSize == sqSumBuf->size<float>());
    float totalSum = std::accumulate(
        sumBuf->data<float>(), sumBuf->data<float>() + totalSize, 0.0);
    float totalSqSum = std::accumulate(
        sqSumBuf->data<float>(), sqSumBuf->data<float>() + totalSize, 0.0);

    float mean = totalSum / (totalSize * featureSize_);
    float stddev =
        std::sqrt(totalSqSum / (totalSize * featureSize_) - mean * mean);

    if (stddev <= kEpsilon) {
      stddev = 1.0;
    }
    meanNormalize(inPtr, featureSize_, mean, stddev, 1.0, 0.0, outPtr);

    if (totalSize > leftContextSize_) {
      sumBuf->consume<float>(1);
      sqSumBuf->consume<float>(1);
    }
    inputBuf->consume<float>(featureSize_);
    outputBuf->move<float>(featureSize_);
  }
  return output;
}

std::string LocalNorm::debugString() const {
  std::stringstream ss;
  ss << "LocalNorm:{featureSize=" << featureSize_
     << " leftContextSize=" << leftContextSize_
     << " rightContextSize=" << rightContextSize_ << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
