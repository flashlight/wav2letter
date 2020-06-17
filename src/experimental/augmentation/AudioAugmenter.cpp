/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "AudioAugmenter.h"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

// TODO: re-enable speedAug() after adding libsamplerate cmake support.
// #include <samplerate.h>

namespace {
constexpr const int kBufferFrames = 1024;
}

namespace w2l {

// std::vector<float>
// speedAug(const std::vector<float>& input, double speed, int channels) {
//   long numInFrames = input.size() / channels;
//   SRC_DATA data;
//   data.data_in = input.data();
//   data.input_frames = numInFrames;
//   data.src_ratio = 1.0 / speed;

//   if (src_is_valid_ratio(data.src_ratio) == 0) {
//     throw std::invalid_argument(
//         "speed factor out of valid range. Valid range: [1.0/256; 256.0]");
//   }

//   long numOutFrames = kBufferFrames + std::ceil(numInFrames *
//   data.src_ratio);
//   std::vector<float> output(numOutFrames * channels);

//   data.data_out = output.data();
//   data.output_frames = numOutFrames;
//   int error = src_simple(&data, SRC_SINC_BEST_QUALITY, channels);
//   if (error != 0) {
//     throw std::invalid_argument(src_strerror(error));
//   }
//   output.resize(data.output_frames_gen * channels);
//   return output;
// }

namespace augmentation {

AudioAndStats::AudioAndStats() : sqrMin_(0.0), sqrMax_(0.0), sqrAvg_(0.0) {}

AudioAndStats::AudioAndStats(std::vector<float> data)
    : sqrMin_(0.0), sqrMax_(0.0), sqrAvg_(0.0), data_(std::move(data)) {
  for (float f : data_) {
    float cur = f * f;
    sqrMin_ = std::min(cur, sqrMin_);
    sqrMax_ = std::min(cur, sqrMax_);
    sqrSum_ += cur;
  }
  sqrAvg_ = sqrSum_ / static_cast<double>(data_.size());
}

std::string AudioAndStats::prettyString() const {
  std::stringstream ss;
  ss << "sqrMin_=" << sqrMin_ << " sqrMax=" << sqrMax_ << " sqrAvg_=" << sqrAvg_
     << " sqrSum_=" << sqrSum_ << " data_.size()=" << data_.size();
  return ss.str();
}

AudioAndStats sumAudiosAndCalcStats(
    std::vector<AudioLoader::Audio> audios,
    size_t len) {
  if (audios.empty()) {
    return {};
  }
  std::vector<float> audioSum = std::move(audios[0].data_);

  // Combine all data into single vector.
  // To maximise chache hits, add one continues memory vector at a time.
  for (int i = 1; i < audios.size(); ++i) {
    std::vector<float>& cur = audios[i].data_;
    for (int j = 0; j < len; ++j) {
      audioSum[j] += cur[j];
    }
  }

  return AudioAndStats(std::move(audioSum));
}

} // namespace augmentation
} // namespace w2l
