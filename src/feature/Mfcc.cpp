/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Mfcc.h"

#include <glog/logging.h>

#include "SpeechUtils.h"

namespace speech {

template <typename T>
Mfcc<T>::Mfcc(const FeatureParams& params)
    : Mfsc<T>(params),
      dct_(params.numFilterbankChans, params.numCepstralCoeffs),
      ceplifter_(params.numCepstralCoeffs, params.lifterParam),
      derivatives_(params.deltaWindow, params.accWindow) {
  validateMfccParams();
}

template <typename T>
std::vector<T> Mfcc<T>::apply(const std::vector<T>& input) {
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
      energy[f] =
          std::log(std::inner_product(begin, begin + nSamples, begin, 0.0));
    }
  }
  auto mfscfeat = this->mfscImpl(frames);
  auto cep = dct_.apply(mfscfeat);
  ceplifter_.applyInPlace(cep);

  auto nFeat = this->featParams_.numCepstralCoeffs;
  if (this->featParams_.useEnergy) {
    if (!this->featParams_.rawEnergy) {
      for (size_t f = 0; f < nFrames; ++f) {
        auto begin = frames.data() + f * nSamples;
        energy[f] =
            std::log(std::inner_product(begin, begin + nSamples, begin, 0.0));
      }
    }
    // Replace C0 with energy
    for (size_t f = 0; f < nFrames; ++f) {
      cep[f * nFeat] = energy[f];
    }
  }
  return derivatives_.apply(cep, nFeat);
}

template <typename T>
int64_t Mfcc<T>::outputSize(int64_t inputSz) {
  return this->featParams_.mfccFeatSz() * this->featParams_.numFrames(inputSz);
}

template <typename T>
void Mfcc<T>::validateMfccParams() const {
  this->validatePowSpecParams();
  this->validateMfscParams();
  LOG_IF(FATAL, this->featParams_.lifterParam < 0)
      << "'lifterparam' has to be >=0.";
}

template class Mfcc<float>;
template class Mfcc<double>;
} // namespace speech
