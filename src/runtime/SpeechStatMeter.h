/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

struct SpeechStats {
  int64_t totalInputSz_;
  int64_t totalTargetSz_;
  int64_t maxInputSz_;
  int64_t maxTargetSz_;
  int64_t numSamples_;

  SpeechStats();
  void reset();
  std::vector<int64_t> toArray();
};

class SpeechStatMeter {
 public:
  SpeechStatMeter();
  void add(const af::array& input, const af::array& target);
  void add(const SpeechStats& stats);
  std::vector<int64_t> value();
  void reset();

 private:
  SpeechStats stats_;
};
} // namespace w2l
