/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <mutex>

#include <fftw3.h>

#include "Dither.h"
#include "FeatureParams.h"
#include "PreEmphasis.h"
#include "Windowing.h"

namespace w2l {

// Computes Power Spectrum features for a speech signal.
template <typename T>
class PowerSpectrum {
 public:
  explicit PowerSpectrum(const FeatureParams& params);

  virtual ~PowerSpectrum();

  // input - input speech signal (T)
  // Returns - Power spectrum (Col Major : FEAT X FRAMESZ)
  virtual std::vector<T> apply(const std::vector<T>& input);

  // input - input speech signal (Col Major : T X BATCHSZ)
  // Returns - Output features (Col Major : FEAT X FRAMESZ X BATCHSZ)
  std::vector<T> batchApply(const std::vector<T>& input, int64_t batchSz);

  virtual int64_t outputSize(int64_t inputSz);

  FeatureParams getFeatureParams() const;

 protected:
  FeatureParams featParams_;

  // Helper function which takes input as signal after dividing the signal into
  // frames. Main purpose of this function is to reuse it in MFSC, MFCC code
  std::vector<T> powSpectrumImpl(std::vector<T>& frames);

  void validatePowSpecParams() const;

 private:
  // The following classes are defined in the order they are applied
  Dither<T> dither_;
  PreEmphasis<T> preEmphasis_;
  Windowing<T> windowing_;

  fftw_plan fftPlan_;
  std::vector<double> inFftBuf_, outFftBuf_;
  std::mutex fftMutex_;
};
} // namespace w2l
