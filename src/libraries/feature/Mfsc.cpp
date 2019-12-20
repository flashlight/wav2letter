/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Mfsc.h"

#include <algorithm>
#include <cstddef>
#include <numeric>

#include "SpeechUtils.h"

namespace w2l {

Mfsc::Mfsc(const FeatureParams& params)
    : PowerSpectrum(params),
      triFltBank_(
          params.numFilterbankChans,
          params.filterFreqResponseLen(),
          params.samplingFreq,
          params.lowFreqFilterbank,
          params.highFreqFilterbank,
          FrequencyScale::MEL),
      derivatives_(params.deltaWindow, params.accWindow) {
  validateMfscParams();
}

std::vector<float> Mfsc::apply(const std::vector<float>& input) {
  auto frames = frameSignal(input, this->featParams_);
  if (frames.empty()) {
    return {};
  }

  int nSamples = this->featParams_.numFrameSizeSamples();
  int nFrames = frames.size() / nSamples;

  std::vector<float> energy(nFrames);
  if (this->featParams_.useEnergy && this->featParams_.rawEnergy) {
    for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      energy[f] = std::log(std::max(
          std::inner_product(
              begin, begin + nSamples, begin, static_cast<float>(0.0)),
          std::numeric_limits<float>::lowest()));
    }
  }
  auto mfscFeat = mfscImpl(frames);
  auto numFeat = this->featParams_.numFilterbankChans;
  if (this->featParams_.useEnergy) {
    if (!this->featParams_.rawEnergy) {
      for (size_t f = 0; f < nFrames; ++f) {
        auto begin = frames.data() + f * nSamples;
        energy[f] = std::log(std::max(
            std::inner_product(
                begin, begin + nSamples, begin, static_cast<float>(0.0)),
            std::numeric_limits<float>::lowest()));
      }
    }
    std::vector<float> newMfscFeat(mfscFeat.size() + nFrames);
    for (size_t f = 0; f < nFrames; ++f) {
      size_t start = f * numFeat;
      newMfscFeat[start + f] = energy[f];
      std::copy(
          mfscFeat.data() + start,
          mfscFeat.data() + start + numFeat,
          newMfscFeat.data() + start + f + 1);
    }
    std::swap(mfscFeat, newMfscFeat);
    ++numFeat;
  }
  // Derivatives will not be computed if windowsize < 0
  return derivatives_.apply(mfscFeat, numFeat);
}

std::vector<float> Mfsc::mfscImpl(std::vector<float>& frames) {
  auto powspectrum = this->powSpectrumImpl(frames);
  if (this->featParams_.usePower) {
    std::transform(
        powspectrum.begin(),
        powspectrum.end(),
        powspectrum.begin(),
        [](float x) { return x * x; });
  }
  auto triflt = triFltBank_.apply(powspectrum, this->featParams_.melFloor);
  std::transform(triflt.begin(), triflt.end(), triflt.begin(), [](float x) {
    return std::log(x);
  });
  return triflt;
}

int Mfsc::outputSize(int inputSz) {
  return this->featParams_.mfscFeatSz() * this->featParams_.numFrames(inputSz);
}

void Mfsc::validateMfscParams() const {
  this->validatePowSpecParams();
  if (this->featParams_.numFilterbankChans <= 0) {
    throw std::invalid_argument("Mfsc: numFilterbankChans must be positive");
  } else if (this->featParams_.melFloor <= 0.0) {
    throw std::invalid_argument("Mfsc: melfloor must be positive");
  }
}
} // namespace w2l
