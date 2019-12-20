/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "Derivatives.h"
#include "FeatureParams.h"
#include "PowerSpectrum.h"
#include "SpeechUtils.h"
#include "TriFilterbank.h"

namespace w2l {

// Computes MFSC features for a speech signal.

class Mfsc : public PowerSpectrum {
 public:
  explicit Mfsc(const FeatureParams& params);

  virtual ~Mfsc() {}

  // input - input speech signal (T)
  // Returns - MFSC feature (Col Major : FEAT X FRAMESZ)
  std::vector<float> apply(const std::vector<float>& input) override;

  int outputSize(int inputSz) override;

 protected:
  // Helper function which takes input as signal after dividing the signal into
  // frames. Main purpose of this function is to reuse it in MFCC code
  std::vector<float> mfscImpl(std::vector<float>& frames);
  void validateMfscParams() const;

 private:
  TriFilterbank triFltBank_;
  Derivatives derivatives_;
};
} // namespace w2l
