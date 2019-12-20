/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Mfcc.h"

#include <cstddef>

#include "SpeechUtils.h"

namespace w2l {

Mfcc::Mfcc(const FeatureParams& params)
    : Mfsc(params),
      dct_(params.numFilterbankChans, params.numCepstralCoeffs),
      ceplifter_(params.numCepstralCoeffs, params.lifterParam),
      derivatives_(params.deltaWindow, params.accWindow) {
  validateMfccParams();
}

std::vector<float> Mfcc::apply(const std::vector<float>& input) {
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

int Mfcc::outputSize(int inputSz) {
  return this->featParams_.mfccFeatSz() * this->featParams_.numFrames(inputSz);
}

void Mfcc::validateMfccParams() const {
  this->validatePowSpecParams();
  this->validateMfscParams();
  if (this->featParams_.lifterParam < 0) {
    throw std::invalid_argument("Mfcc: lifterparam must be nonnegative");
  }
}

} // namespace w2l
