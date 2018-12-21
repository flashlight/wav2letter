/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Mfsc.h"

#include <algorithm>
#include <numeric>

#include <glog/logging.h>

#include "SpeechUtils.h"

namespace speech {

template <typename T>
Mfsc<T>::Mfsc(const FeatureParams& params)
    : PowerSpectrum<T>(params),
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

template <typename T>
std::vector<T> Mfsc<T>::apply(const std::vector<T>& input) {
  auto frames = frameSignal(input, this->featParams_);
  if (frames.empty()) {
    return {};
  }

  int64_t nSamples = this->featParams_.numFrameSizeSamples();
  int64_t nFrames = frames.size() / nSamples;

  std::vector<T> energy(nFrames);
  if (this->featParams_.useEnergy && this->featParams_.rawEnergy) {
    for (size_t f = 0; f < nFrames; ++f) {
      auto begin = frames.data() + f * nSamples;
      energy[f] = std::log(std::max(
          std::inner_product(
              begin, begin + nSamples, begin, static_cast<T>(0.0)),
          std::numeric_limits<T>::min()));
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
                begin, begin + nSamples, begin, static_cast<T>(0.0)),
            std::numeric_limits<T>::min()));
      }
    }
    std::vector<T> newMfscFeat(mfscFeat.size() + nFrames);
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

template <typename T>
std::vector<T> Mfsc<T>::mfscImpl(std::vector<T>& frames) {
  auto powspectrum = this->powSpectrumImpl(frames);
  if (this->featParams_.usePower) {
    std::transform(
        powspectrum.begin(), powspectrum.end(), powspectrum.begin(), [](T x) {
          return x * x;
        });
  }
  auto triflt = triFltBank_.apply(powspectrum, this->featParams_.melFloor);
  std::transform(triflt.begin(), triflt.end(), triflt.begin(), [](T x) {
    return std::log(x);
  });
  return triflt;
}

template <typename T>
int64_t Mfsc<T>::outputSize(int64_t inputSz) {
  return this->featParams_.mfscFeatSz() * this->featParams_.numFrames(inputSz);
}

template <typename T>
void Mfsc<T>::validateMfscParams() const {
  this->validatePowSpecParams();
  LOG_IF(FATAL, this->featParams_.numFilterbankChans <= 0)
      << "numfilterbankchans' has to be positive.";
  LOG_IF(FATAL, this->featParams_.melFloor <= 0.0)
      << "'melfloor' has to be positive.";
}

template class Mfsc<float>;
template class Mfsc<double>;
} // namespace speech
