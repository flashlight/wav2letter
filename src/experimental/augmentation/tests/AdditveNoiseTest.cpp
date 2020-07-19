/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

// #include "data/Sound.h"
#include "experimental/augmentation/AdditiveNoise.h"
// #include "experimental/augmentation/AudioLoader.h"
// #include "libraries/common/Utils.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

const std::string kNoiseInputDirectory =
    "/checkpoint/avidov/experiments/noise/audio/noise";
// const std::string kSignalInputDirectory =
//     "/checkpoint/antares/datasets/librispeech/audio/LibriSpeech/train-clean-100/911/130578";
const std::string kSignalInputDirectory = "/private/home/avidov/signal";

const std::vector<std::string> signals = {
    "/checkpoint/antares/datasets/librispeech/audio/LibriSpeech/train-clean-100/911/130578/911-130578-0016.flac",
    "/checkpoint/antares/datasets/librispeech/audio/LibriSpeech/train-clean-100/911/130578/911-130578-0005.flac"};

const std::vector<std::string> noises = {
    "/private/home/avidov/noise/noise_fileid_7771.wav",
    "/private/home/avidov/noise/noise_fileid_7772.wav",
    "/private/home/avidov/noise/noise_fileid_7773.wav",
    "/private/home/avidov/noise/noise_fileid_7774.wav",
    "/private/home/avidov/noise/noise_fileid_7775.wav",
};

TEST(AdditiveNoise, AudioLoader) {
  augmentation::AudioLoader noiseDb(kNoiseInputDirectory);
  augmentation::AudioLoader::Audio noise = noiseDb.loadRandom();
  EXPECT_GT(noise.data_.size(), 0);
  std::cout << noise.prettyString() << std::endl;
}

TEST(AdditiveNoise, LogAugmentedSamplesWithVariousfConfigs) {
  augmentation::SoundEffect::Config sfxConfig;
  sfxConfig.debug_.debugLevel_ =
      0; // 3; // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
  sfxConfig.debug_.outputPath_ = "/tmp";
  sfxConfig.debug_.outputFilePrefix_ = "additive-noise-";

  augmentation::AudioLoader signalDb(kSignalInputDirectory);
  const std::vector<augmentation::AudioLoader::Audio> signals = {
      signalDb.loadRandom()};

  for (const augmentation::AudioLoader::Audio& signal : signals) {
    for (int nClipsPerUtterance_ = 1; nClipsPerUtterance_ <= 5;
         ++nClipsPerUtterance_) {
      for (double minSnr = 0.125; minSnr <= 40.0; minSnr *= 2.0) {
        for (double maxTimeRatio_ = 0; maxTimeRatio_ <= 1;
             maxTimeRatio_ += 0.25) {
          augmentation::AdditiveNoise::Config noiseConfig;
          noiseConfig.maxTimeRatio_ = maxTimeRatio_;
          noiseConfig.minSnr_ = minSnr;
          noiseConfig.maxSnr_ = minSnr * 2;
          noiseConfig.nClipsPerUtterance_ = nClipsPerUtterance_;
          noiseConfig.noiseDir_ = kNoiseInputDirectory;

          augmentation::AdditiveNoise noiseAdder(sfxConfig, noiseConfig);
          std::vector<float> augmented = signal.data_;
          noiseAdder(&augmented);

          EXPECT_EQ(augmented.size(), signal.data_.size());
          std::cout << "minSnr=" << minSnr << std::endl;
        }
      }
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
