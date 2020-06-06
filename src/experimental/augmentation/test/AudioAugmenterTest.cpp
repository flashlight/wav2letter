/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "data/Sound.h"
// #include "experimental/augmentation/AudioAugmenter.h"
#include "experimental/augmentation/AdditiveNoise.h"
#include "experimental/augmentation/NoiseDatabase.h"
#include "libraries/common/Utils.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

// TEST(AudioAugmenterTest, SpeedAug) {
//   std::string filename = "test.wav";
//   std::string fullpath = loadPath + "/" + filename;
//   auto data = loadSound<float>(fullpath);
//   auto info = loadSoundInfo(fullpath);
//   ASSERT_EQ(data.size(), info.frames * info.channels);

//   std::vector<double> speeds = {0.8, 1.2};
//   for (const auto& speed : speeds) {
//     auto output = speedAug(data, speed, info.channels);
//     ASSERT_NEAR(output.size(), data.size() / speed, 10);
//     saveSound(
//         "/tmp/" + std::to_string(speed) + "x_" + filename,
//         output,
//         info.samplerate,
//         info.channels,
//         SoundFormat::WAV,
//         SoundSubFormat::PCM_16);
//   }
// }

const std::string kNoiseInputDirectory = "/home/avidov/noise";

TEST(AudioAugmenterTest, NoiseDatabase) {
  w2l::augmentation::NoiseDatabase noiseDb(kNoiseInputDirectory);
  const size_t numTracks = 3;
  std::vector<std::vector<float>> noiseVec =
      noiseDb.getRandomNoiseTracks(numTracks);
  EXPECT_EQ(noiseVec.size(), numTracks);
}

TEST(AudioAugmenterTest, AdditiveNoise) {
  w2l::augmentation::AdditiveNoise noiseAdder(kNoiseInputDirectory, {});
  const size_t signalSize = 1000;
  std::vector<float> signal(signalSize);
  for (int i = 0; i < signalSize; ++i) {
    signal[i] = 0;
  }

  const std::vector<float> augmented = noiseAdder.ApplyAdditiveNoise(signal);

  EXPECT_NE(augmented, signal);
}

// TEST(AudioAugmenterTest, Mix) {
//   // std::string filename = "test.wav";
//   // std::string fullpath = loadPath + "/" + filename;
//   const std::string inputSignalFilename = "noise_fileid_7773.wav";
//   const std::string inputNoiseFilename = "noise_fileid_7778.wav";
//   const std::string outputFilename =
//       inputSignalFilename + "-augmented-with-noise-" + inputNoiseFilename;

//   const std::string fullpath =
//       w2l::pathsConcat(kNoiseInputDirectory, outputFilename);

//   SoundSample signal = loadSoundSample(
//       w2l::pathsConcat(kNoiseInputDirectory, inputSignalFilename));
//   SoundSample noise = loadSoundSample(
//       w2l::pathsConcat(kNoiseInputDirectory, inputNoiseFilename));
//   SoundSample output = Mix(signal, noise, {});

//   saveSound(
//       w2l::pathsConcat(kNoiseInputDirectory, outputFilename),
//       output.data,
//       signal.info.samplerate,
//       signal.info.channels,
//       SoundFormat::WAV,
//       SoundSubFormat::PCM_16);
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
