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

template <typename T>
class TriFilterbank {
 public:
  TriFilterbank(
      int64_t numfilters,
      int64_t filterlen,
      int64_t samplingfreq,
      int64_t lowfreq = 0,
      int64_t highfreq = -1,
      FrequencyScale freqscale = FrequencyScale::MEL);

  std::vector<T> apply(const std::vector<T>& input, T melfloor = 0.0) const;

  // Returns triangular filterbank matrix
  std::vector<T> filterbank() const;

 private:
  int64_t numFilters_; // Number of filterbank channels
  int64_t filterLen_; // length of each filterbank channel
  int64_t samplingFreq_; // sampling frequency (Hz)
  int64_t lowFreq_; // lower cutoff frequency (Hz)
  int64_t highFreq_; // higher cutoff frequency (Hz)
  FrequencyScale freqScale_; // frequency warp type Ex. FrequencyScale::MEL
  std::vector<T> H_; // (numFilters_ x filterLen_) triangular filterbank matrix

  T hertzToWarpedScale(T hz, FrequencyScale freqscale) const;
  T warpedToHertzScale(T wrp, FrequencyScale freqscale) const;
};
} // namespace w2l
