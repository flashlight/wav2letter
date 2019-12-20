/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

#include "FeatureParams.h"

namespace w2l {

class TriFilterbank {
 public:
  TriFilterbank(
      int numfilters,
      int filterlen,
      int samplingfreq,
      int lowfreq = 0,
      int highfreq = -1,
      FrequencyScale freqscale = FrequencyScale::MEL);

  std::vector<float> apply(
      const std::vector<float>& input,
      float melfloor = 0.0) const;

  // Returns triangular filterbank matrix
  std::vector<float> filterbank() const;

 private:
  int numFilters_; // Number of filterbank channels
  int filterLen_; // length of each filterbank channel
  int samplingFreq_; // sampling frequency (Hz)
  int lowFreq_; // lower cutoff frequency (Hz)
  int highFreq_; // higher cutoff frequency (Hz)
  FrequencyScale freqScale_; // frequency warp type Ex. FrequencyScale::MEL
  std::vector<float>
      H_; // (numFilters_ x filterLen_) triangular filterbank matrix

  float hertzToWarpedScale(float hz, FrequencyScale freqscale) const;
  float warpedToHertzScale(float wrp, FrequencyScale freqscale) const;
};
} // namespace w2l
