/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/Sound.h"
#include "experimental/augmentation/AdditiveNoise.h"
#include "experimental/augmentation/AdditiveNoise.h"
#include "experimental/augmentation/NoiseDatabase.h"
#include "libraries/common/Utils.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

const std::string kNoiseInputDirectory =
    "/checkpoint/avidov/experiments/noise/audio/noise";
const std::string kSignalInputDirectory =
    "/checkpoint/antares/datasets/librispeech/audio/LibriSpeech/train-clean-100/911/130578";

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

TEST(AudioAugmenterTest, NoiseDatabase) {
  w2l::augmentation::NoiseDatabase noiseDb(kNoiseInputDirectory);
  const size_t numTracks = 3;
  std::vector<std::vector<float>> noiseVec =
      noiseDb.getRandomNoiseTracks(numTracks);
  EXPECT_EQ(noiseVec.size(), numTracks);
}

TEST(AudioAugmenterTest, AdditiveNoise) {
  w2l::augmentation::NoiseDatabase signalDb(kSignalInputDirectory);
  const size_t numTracks = 1;
  std::vector<std::vector<float>> signals =
      signalDb.getRandomNoiseTracks(numTracks);

  w2l::augmentation::AdditiveNoise::Config config;
  config.randomSeed_ = 0;
  config.maxTimeRatio_ = 1;
  config.minSnr_ = 1;
  config.maxSnr_ = 1;
  config.nClipsPerUtterance_ = 2;
  config.noiseDir_ = kNoiseInputDirectory;
  config.debugLevel_ =
      3; // 0=none, 1=stats, 2=histogram, 3=saveq augmented files

  w2l::augmentation::AdditiveNoise noiseAdder(config);

  const std::vector<float> augmented =
      noiseAdder.ApplyAdditiveNoise(signals[0]);

  EXPECT_NE(augmented, signal);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
