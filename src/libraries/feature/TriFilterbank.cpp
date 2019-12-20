/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "TriFilterbank.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "SpeechUtils.h"

namespace w2l {

TriFilterbank::TriFilterbank(
    int numfilters,
    int filterlen,
    int samplingfreq,
    int lowfreq /* = 0 */,
    int highfreq /* = -1 */,
    FrequencyScale freqscale /* = FrequencyScale::MEL */)
    : numFilters_(numfilters),
      filterLen_(filterlen),
      samplingFreq_(samplingfreq),
      lowFreq_(lowfreq),
      highFreq_((highfreq > 0) ? highfreq : (samplingfreq >> 1)),
      freqScale_(freqscale),
      H_(filterlen * numfilters) {
  float minwarpfreq = hertzToWarpedScale(lowFreq_, freqScale_);
  float maxwarpfreq = hertzToWarpedScale(highFreq_, freqScale_);
  float dwarp = (maxwarpfreq - minwarpfreq) / (numfilters + 1);

  std::vector<float> f(numFilters_ + 2);
  for (int i = 0; i < (numFilters_ + 2); ++i) {
    f[i] = warpedToHertzScale(i * dwarp + minwarpfreq, freqScale_) *
        (filterLen_ - 1) * 2.0 / samplingFreq_;
  }

  float minH = 0.0;

  for (size_t i = 0; i < filterLen_; ++i) {
    for (size_t j = 0; j < numFilters_; ++j) {
      float hislope = (i - f[j]) / (f[j + 1] - f[j]);
      float loslope = (f[j + 2] - i) / (f[j + 2] - f[j + 1]);
      H_[i * numFilters_ + j] = std::max(std::min(hislope, loslope), minH);
    }
  }
}

std::vector<float> TriFilterbank::apply(
    const std::vector<float>& input,
    float melfloor /* = 0.0 */) const {
  std::vector<float> output = cblasGemm(input, H_, numFilters_, filterLen_);
  std::transform(
      output.begin(),
      output.end(),
      output.begin(),
      [melfloor](float n) -> float { return std::max(n, melfloor); });
  return output;
}

std::vector<float> TriFilterbank::filterbank() const {
  return H_;
}

float TriFilterbank::hertzToWarpedScale(float hz, FrequencyScale freqscale)
    const {
  switch (freqscale) {
    case FrequencyScale::MEL:
      return 2595.0 * log10(1.0 + hz / 700.0);
    case FrequencyScale::LOG10:
      return log10(hz);
    case FrequencyScale::LINEAR:
      return hz;
    default:
      throw std::invalid_argument("TriFilterbank: unsupported frequency scale");
      return 0.0;
  }
}

float TriFilterbank::warpedToHertzScale(float wrp, FrequencyScale freqscale)
    const {
  switch (freqscale) {
    case FrequencyScale::MEL:
      return 700.0 * (pow(10, wrp / 2595.0) - 1);
    case FrequencyScale::LOG10:
      return pow(10, wrp);
    case FrequencyScale::LINEAR:
      return wrp;
    default:
      throw std::invalid_argument("TriFilterbank: unsupported frequency scale");
      return 0.0;
  }
}
} // namespace w2l
