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

namespace speech {

// Computes MFSC features for a speech signal.

template <typename T>
class Mfsc : public PowerSpectrum<T> {
 public:
  explicit Mfsc(const FeatureParams& params);

  virtual ~Mfsc() {}

  // input - input speech signal (T)
  // Returns - MFSC feature (Col Major : FEAT X FRAMESZ)
  std::vector<T> apply(const std::vector<T>& input) override;

  int64_t outputSize(int64_t inputSz) override;

 protected:
  // Helper function which takes input as signal after dividing the signal into
  // frames. Main purpose of this function is to reuse it in MFCC code
  std::vector<T> mfscImpl(std::vector<T>& frames);
  void validateMfscParams() const;

 private:
  TriFilterbank<T> triFltBank_;
  Derivatives<T> derivatives_;
};
} // namespace speech
