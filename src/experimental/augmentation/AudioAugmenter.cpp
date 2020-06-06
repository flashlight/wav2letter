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

AudioAndStats::AudioAndStats() : absMin_(0.0), absMax_(0.0), absAvg_(0.0) {}

std::string AudioAndStats::prettyString() {
  std::stringstream ss;
  ss << "absMin_=" << absMin_ << " absMax=" << absMax_ << " absAvg_=" << absAvg_
     << " absSum_=" << absSum_ << " data_.size()=" << data_.size();
  return ss.str();
}

AudioAndStats sumAudiosAndCalcStats(
    std::vector<AudioLoader::Audio> audios,
    size_t len) {
  if (audios.empty()) {
    return {};
  }
  AudioAndStats stats;
  stats.data_ = std::move(audios[0].data_);
  stats.data_.resize(len, 0.0);

  // Combine all data into single vector.
  // To maximise chache hits, add one continues memory vector at a time.
  for (int i = 1; i < audios.size(); ++i) {
    std::vector<float>& data = audios[i].data_;
    for (int j = 0; j < len; ++j) {
      stats.data_[j] += data[j];
    }
  }

  for (int i = 0; i < len; ++i) {
    float cur = std::abs(stats.data_[i]);
    stats.absMin_ = std::min(cur, stats.absMin_);
    stats.absMax_ = std::min(cur, stats.absMax_);
    stats.absSum_ += cur;
  }
  stats.absAvg_ = stats.absSum_ / static_cast<double>(len);
  return stats;
}

AudioAndStats calcAudioStats(std::vector<float> audio) {
  if (audio.empty()) {
    return {};
  }
  AudioAndStats stats;
  stats.data_ = std::move(audio);

  for (int i = 0; i < stats.data_.size(); ++i) {
    float cur = std::abs(stats.data_[i]);
    stats.absMin_ = std::min(cur, stats.absMin_);
    stats.absMax_ = std::min(cur, stats.absMax_);
    stats.absSum_ += cur;
  }
  stats.absAvg_ = stats.absSum_ / static_cast<double>(stats.data_.size());
  return stats;
}

AudioAndStats calcAudioAndStats(
    std::vector<std::vector<float>> audios,
    size_t len) {
  if (audios.empty()) {
    return {};
  }
  AudioAndStats stats;
  stats.data_ = std::move(audios[0]);
  stats.data_.resize(len, 0.0);

  // Combine all data into single vector.
  // To maximise chache hits, add one continues memory vector at a time.
  for (int i = 1; i < audios.size(); ++i) {
    for (int j = 0; j < len; ++j) {
      stats.data_[j] += audios[i][j];
    }
  }

  for (int i = 0; i < len; ++i) {
    float cur = std::abs(stats.data_[i]);
    stats.absMin_ = std::min(cur, stats.absMin_);
    stats.absMax_ = std::min(cur, stats.absMax_);
    stats.absSum_ += cur;
  }
  stats.absAvg_ = stats.absSum_ / static_cast<double>(len);
  return stats;
}

} // namespace augmentation
} // namespace w2l
