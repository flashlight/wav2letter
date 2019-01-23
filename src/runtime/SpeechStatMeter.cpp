/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SpeechStatMeter.h"
#include <array>

namespace w2l {
SpeechStatMeter::SpeechStatMeter() {
  reset();
}

void SpeechStatMeter::reset() {
  stats_.reset();
}

void SpeechStatMeter::add(const af::array& input, const af::array& target) {
  int64_t curInputSz = input.dims(0);
  int64_t curTargetSz = target.dims(0);

  stats_.totalInputSz_ += curInputSz;
  stats_.totalTargetSz_ += curTargetSz;

  stats_.maxInputSz_ = std::max(stats_.maxInputSz_, curInputSz);
  stats_.maxTargetSz_ = std::max(stats_.maxTargetSz_, curTargetSz);

  stats_.numSamples_ += 1;
}

void SpeechStatMeter::add(const SpeechStats& stats) {
  stats_.totalInputSz_ += stats.totalInputSz_;
  stats_.totalTargetSz_ += stats.totalTargetSz_;

  stats_.maxInputSz_ = std::max(stats_.maxInputSz_, stats.maxInputSz_);
  stats_.maxTargetSz_ = std::max(stats_.maxTargetSz_, stats.maxTargetSz_);

  stats_.numSamples_ += stats.numSamples_;
}

std::vector<int64_t> SpeechStatMeter::value() {
  return stats_.toArray();
}

SpeechStats::SpeechStats() {
  reset();
}

void SpeechStats::reset() {
  totalInputSz_ = 0;
  totalTargetSz_ = 0;
  maxInputSz_ = 0;
  maxTargetSz_ = 0;
  numSamples_ = 0;
}

std::vector<int64_t> SpeechStats::toArray() {
  std::vector<int64_t> arr(5);
  arr[0] = totalInputSz_;
  arr[1] = totalTargetSz_;
  arr[2] = maxInputSz_;
  arr[3] = maxTargetSz_;
  arr[4] = numSamples_;
  return arr;
}

} // namespace w2l
