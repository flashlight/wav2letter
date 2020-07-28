/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <chrono>
#include <sstream>
#include <string>

#include "data/Sound.h"
#include "experimental/augmentation/Reverberation.h"
// #include "experimental/augmentation/AudioLoader.h"
// #include "libraries/common/Utils.h"
// #include "flashlight/nn/modules/Conv2D.h"
#include "flashlight/autograd/Functions.h"
// #include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/nn/Init.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

const std::string signalFile =
    "/datasets01/librispeech/062419/train-clean-100/87/121553/87-121553-0055.flac";
const std::string kSignalInputDirectory = "/private/home/avidov/signal";
const float kPI = 3.14;
const float absorptionCoefficientMin = 0.05; /*painted brick*/
const float absorptionCoefficientMax = 1.5; /* same as original */
const float distanceToWallInMetersMin = 0.1;
const float distanceToWallInMetersMax = 100.0;
const size_t numWallsMin = 1;
const size_t numWallsMax = 32;

std::string prettyDuration(
    const std::chrono::time_point<std::chrono::high_resolution_clock>& start,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& end) {
  const auto runtime = end - start;
  auto runtimeMicroSec =
      std::chrono::duration_cast<std::chrono::microseconds>(runtime);
  auto runtimeMiliSec =
      std::chrono::duration_cast<std::chrono::milliseconds>(runtime);
  auto runtimeSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(runtime);
  std::stringstream strStream;
  strStream << " elapsed time=";
  if (runtimeMicroSec.count() < 1e5) {
    strStream << runtimeMicroSec.count() << " microseconds\n";
  } else if (runtimeMiliSec.count() < 1e5) {
    strStream << runtimeMiliSec.count() << " milliseconds\n";
  } else {
    strStream << runtimeSeconds.count() << " seconds\n";
  }
  return strStream.str();
}

class TimeElapsedReporter {
 public:
  explicit TimeElapsedReporter(std::string name);
  ~TimeElapsedReporter();

 private:
  const std::string name_;
  const std::chrono::time_point<std::chrono::high_resolution_clock>
      startTimepoint_;
};

TimeElapsedReporter::TimeElapsedReporter(std::string name)
    : name_(std::move(name)),
      startTimepoint_(std::chrono::high_resolution_clock::now()) {
  std::cout << "Started " << name_ << " ... " << std::endl;
}

TimeElapsedReporter::~TimeElapsedReporter() {
  std::cout << "Completed " << name_
            << prettyDuration(
                   startTimepoint_, std::chrono::high_resolution_clock::now())
            << std::endl;
}

TEST(Reverberation, Impulse) {
  const int sampleRate = 16000;
  const float lenSecs = 0.01;
  const size_t signalLen = sampleRate * lenSecs;
  std::vector<float> signal(signalLen, 0);
  signal[0] = 1.0;

  augmentation::SoundEffect::Config sfxConfig;
  // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
  sfxConfig.debug_.debugLevel_ = 3;
  sfxConfig.debug_.outputPath_ = "/tmp";
  sfxConfig.debug_.outputFilePrefix_ = "reverb-speech-matrix-samples";

  augmentation::Reverberation::Config reverbConfig;
  reverbConfig.numWallsMin_ = 1;
  reverbConfig.numWallsMax_ = 1;
  reverbConfig.jitter_ = 0;

  float signalSampleDistance =
      augmentation::kSpeedOfSoundMeterPerSec / sampleRate;
  std::vector<float> augmented;

  for (int b = static_cast<int>(
           augmentation::Reverberation::Config::Backend::GPU_GAB);
       b <=
       static_cast<int>(augmentation::Reverberation::Config::Backend::GPU_COV);
       b++) {
    reverbConfig.backend_ =
        static_cast<augmentation::Reverberation::Config::Backend>(b);
    for (float d = 1.0; d <= 100.0; d *= 5) {
      reverbConfig.distanceToWallInMetersMin_ = d * signalSampleDistance;
      reverbConfig.distanceToWallInMetersMax_ = d * signalSampleDistance;

      for (float a = 0.000001; a <= 1.0; a += 0.5) {
        reverbConfig.absorptionCoefficientMin_ = a;
        reverbConfig.absorptionCoefficientMax_ = a;

        augmentation::Reverberation reveberation(sfxConfig, reverbConfig);
        reveberation.enable(true);

        augmented = signal;
        reveberation(&augmented);
        std::cout << "augmented:" << std::endl;
        for (int i = 0; i < augmented.size(); ++i) {
          std::cout << augmented[i] << ", ";
        }
        std::cout << std::endl;
        EXPECT_EQ(augmented.size(), signal.size());
      }
    }
  }
}

// TEST(Reverberation, SinWavePulses) {
//   const int sampleRate = 16000;
//   const float lenSecs = 4;
//   const size_t signalLen = sampleRate * lenSecs;
//   const float sinWaveAmplitude = 0.2;
//   const float sinWaveFreq = 2000;
//   const float ratio = (2.0 * kPI * sinWaveFreq) / (sampleRate);
//   const float pulseCount = 5;
//   const float fillRatio = 0.1;

//   std::vector<float> signal(signalLen, 0);
//   for (int j = 0; j < pulseCount; ++j) {
//     const int start = j * (signalLen / pulseCount);
//     const int frames = (signalLen * fillRatio) / (pulseCount);
//     for (int i = start; i < start + frames; ++i) {
//       signal[i] = sin(static_cast<float>(i) * ratio) *
//       sinWaveAmplitude;
//     }
//   }

//   augmentation::Reverberation::Config reverbConfig;
//   // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
//   reverbConfig.debugLevel_ = 3;
//   reverbConfig.debugOutputPath_ = "/tmp";
//   reverbConfig.debugOutputFilePrefix_ = "reverb-pulses";
//   reverbConfig.lengthMilliseconds_ = 1;

//   std::vector<float> augmented;
//   for (float d = distanceToWallInMetersMin; d <
//   distanceToWallInMetersMax;
//        d *= 4) {
//     reverbConfig.distanceToWallInMetersMin_ = d;
//     reverbConfig.distanceToWallInMetersMax_ = d;
//     for (int e = numWallsMin; e <= numWallsMax; e *= 4) {
//       reverbConfig.numWallsMin_ = e;
//       reverbConfig.numWallsMax_ = e;
//       for (float a = absorptionCoefficientMin; a <=
//       absorptionCoefficientMax;
//            a *= 4) {
//         reverbConfig.absorptionCoefficientMin_ = a;
//         reverbConfig.absorptionCoefficientMax_ = a;

//         augmentation::Reverberation reveberation(reverbConfig);
//         reveberation.enable(true);

//         augmented = signal;
//         reveberation.augment(&augmented, nullptr);
//         EXPECT_EQ(augmented.size(), signal.size());
//       }
//     }
//   }
// }

TEST(Reverberation, libriSpeecIterateOverConfigs) {
  std::vector<float> signal = w2l::loadSound<float>(signalFile);

  augmentation::SoundEffect::Config sfxConfig;
  // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
  sfxConfig.debug_.debugLevel_ = 3;
  sfxConfig.debug_.outputPath_ = "/tmp";
  sfxConfig.debug_.outputFilePrefix_ = "reverb-speech-matrix-samples";

  augmentation::Reverberation::Config reverbConfig;

  std::vector<float> augmented;
  for (float d = distanceToWallInMetersMin; d < distanceToWallInMetersMax;
       d *= 4) {
    reverbConfig.distanceToWallInMetersMin_ = d;
    reverbConfig.distanceToWallInMetersMax_ = d;

    for (int e = numWallsMin; e <= numWallsMax; e *= 4) {
      reverbConfig.numWallsMin_ = e;
      reverbConfig.numWallsMax_ = e;
      for (float a = absorptionCoefficientMin; a <= absorptionCoefficientMax;
           a *= 4) {
        reverbConfig.absorptionCoefficientMin_ = a;
        reverbConfig.absorptionCoefficientMax_ = a;

        augmentation::Reverberation reveberation(sfxConfig, reverbConfig);
        reveberation.enable(true);

        augmented = signal;
        reveberation(&augmented);
        EXPECT_EQ(augmented.size(), signal.size());
      }
    }
  }
}

// TEST(Reverberation, libriSpeechRandom) {
//   std::vector<float> signal = w2l::loadSound<float>(signalFile);

//   augmentation::SoundEffect::Config sfxConfig;
//   // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
//   sfxConfig.debug_.debugLevel_ = 3;
//   sfxConfig.debug_.outputPath_ = "/tmp";
//   sfxConfig.debug_.outputFilePrefix_ = "reverb-speech-matrix-samples";

//   augmentation::Reverberation::Config reverbConfig;
//   reverbConfig.distanceToWallInMetersMin_ = distanceToWallInMetersMin;
//   reverbConfig.distanceToWallInMetersMax_ = distanceToWallInMetersMax;
//   reverbConfig.numWallsMin_ = numWallsMin;
//   reverbConfig.numWallsMax_ = numWallsMax;
//   reverbConfig.absorptionCoefficientMin_ = absorptionCoefficientMin;
//   reverbConfig.absorptionCoefficientMax_ = absorptionCoefficientMax;

//   augmentation::Reverberation reveberation(sfxConfig, reverbConfig);
//   reveberation.enable(true);

//   std::vector<float> augmented;
//   for (int j = 0; j < 100; ++j) {
//     augmented = signal;
//     {
//       TimeElapsedReporter elapsed("reverb-augmentation");
//       reveberation(&augmented);
//     }
//     EXPECT_EQ(augmented.size(), signal.size());
//   }
// }

// TEST(Reverberation, LogAugmentedSamplesWithVariousfConfigs) {
//   augmentation::AudioLoader signalDb(kSignalInputDirectory);
//   const std::vector<augmentation::AudioLoader::Audio> signals = {
//       signalDb.loadRandom(), signalDb.loadRandom(),
//       signalDb.loadRandom()};

// augmentation::AdditiveNoise::Config sfxConfig;
// // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
// sfxConfig.debug_.debugLevel_ = 3;
// sfxConfig.debug_.debugOutputPath_ = "/tmp";
// sfxConfig.debug_.debugOutputFilePrefix_ =
// "reverb-speech-matrix-samples";

//   augmentation::Reverberation::Config reverbConfig;
//   augmentation::Reverberation reveberation(reverbConfig);
//   reveberation.enable(true);
//   for (const augmentation::AudioLoader::Audio& signal : signals) {
//     std::vector<float> augmented = signal.data_;
//     reveberation.augment(&augmented);

//     EXPECT_EQ(augmented.size(), signal.data_.size());
//   }
// }

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

// Resolve directory for data
#ifdef DATA_TEST_DATADIR
  loadPath = DATA_TEST_DATADIR;
#endif

  return RUN_ALL_TESTS();
}
