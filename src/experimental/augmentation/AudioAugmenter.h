/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include "data/Sound.h"
#include "experimental/augmentation/AudioLoader.h"

namespace w2l {

/**
 * Performs speed pertubation to the audio input by a fixed rate.
 *
 * @param input a float vector of input of shape channels X T (row major)
 * @param speed speed factor to be used
 * @param channels number of channels in the input audio
 * @return a float vector of shape channels X T' where T' ~= T / speed
 */
std::vector<float>
speedAug(const std::vector<float>& input, double speed, int channels = 1);

namespace augmentation {

struct AudioAndStats {
  AudioAndStats();
  explicit AudioAndStats(std::vector<float> data);

  std::string prettyString() const;

  float sqrMin_;
  float sqrMax_;
  float sqrAvg_;
  double sqrSum_;
  std::vector<float> data_;
};

AudioAndStats sumAudiosAndCalcStats(std::vector<AudioLoader::Audio> audios, size_t len);

class AudioAugmenter {
 public:
  virtual ~AudioAugmenter() = default;

  virtual void augment(std::vector<float>& signal) = 0;
};

} // namespace augmentation
} // namespace w2l
