/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/feature/LogMelFeature.h"

#include <cassert>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

LogMelFeature::LogMelFeature()
    : numFilters_(0),
      frameSizeMs_(25),
      frameShiftMs_(10),
      samplingFreq_(16000) {}

LogMelFeature::LogMelFeature(
    int numFilters,
    int frameSizeMs /* = 25 */,
    int frameShiftMs /* = 10 */,
    int samplingFreq /* = 16000 */)
    : numFilters_(numFilters),
      frameSizeMs_(frameSizeMs),
      frameShiftMs_(frameShiftMs),
      samplingFreq_(samplingFreq) {
  init();
  assert(mfscFeaturizer_);
}

std::shared_ptr<ModuleProcessingState> LogMelFeature::start(
    std::shared_ptr<ModuleProcessingState> input) {
  return input->next(true, 1);
}

std::shared_ptr<ModuleProcessingState> LogMelFeature::run(
    std::shared_ptr<ModuleProcessingState> input) {
  assert(input);
  std::shared_ptr<ModuleProcessingState> output = input->next();
  assert(output);
  std::shared_ptr<IOBuffer> inputBuf = input->buffer(0);
  assert(inputBuf);
  int inputSize = inputBuf->size<float>();

  std::shared_ptr<IOBuffer> outputBuf = output->buffer(0);
  assert(outputBuf);

  auto numFrames = featParams_.numFrames(inputSize);
  if (numFrames == 0) {
    return output;
  }
  std::vector<float> inputVec(inputBuf->data<float>(), inputBuf->tail<float>());
  auto outputVec = mfscFeaturizer_->apply(inputVec);

  assert(outputVec.size() == numFrames * numFilters_);
  outputBuf->write<float>(outputVec.data(), outputVec.size());
  inputBuf->consume<float>(numFrames * featParams_.numFrameStrideSamples());
  return output;
}

std::string LogMelFeature::debugString() const {
  std::stringstream ss;
  ss << "LogMelFeature:{numFilters=" << numFilters_
     << " frameSizeMs=" << frameSizeMs_ << " frameShiftMs=" << frameShiftMs_
     << " samplingFreq=" << samplingFreq_ << "}";
  return ss.str();
}

void LogMelFeature::init() {
  featParams_.samplingFreq = samplingFreq_;
  featParams_.frameSizeMs = frameSizeMs_;
  featParams_.frameStrideMs = frameShiftMs_;
  featParams_.lowFreqFilterbank = 0;
  featParams_.highFreqFilterbank = samplingFreq_ / 2;
  featParams_.zeroMeanFrame = false;
  featParams_.ditherVal = 0.0;

  featParams_.numFilterbankChans = numFilters_;
  featParams_.useEnergy = false;
  featParams_.usePower = false;
  featParams_.accWindow = 0;
  featParams_.deltaWindow = 0;

  mfscFeaturizer_ = std::make_shared<Mfsc>(featParams_);
}

} // namespace streaming
} // namespace w2l
