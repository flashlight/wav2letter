/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <sstream>

// #include "data/Sound.h"
#include "experimental/augmentation/Reverberation.h"
// #include "experimental/augmentation/AudioLoader.h"
// #include "libraries/common/Utils.h"
// #include "flashlight/nn/modules/Conv2D.h"
#include "flashlight/autograd/Functions.h"
#include "flashlight/flashlight/autograd/Variable.h"
#include "flashlight/flashlight/nn/Init.h"

using namespace w2l;

namespace {
std::string loadPath = "";
}

const std::string kSignalInputDirectory = "/private/home/avidov/signal";
const float kPI = 3.14;

TEST(Reverberation, FakeSignal) {
  const int sampleRate = 16000;
  const float lenSecs = 4;
  const size_t signalLen = sampleRate * lenSecs;
  const float maxAmplitude = 0.2;
  const float freq = 2000;
  const float ratio = (2.0 * kPI * freq) / (sampleRate);
  const float impulseCnt = 5;
  const float fillRatio = 0.1;

  std::vector<float> signal(signalLen, 0);

  for (int j = 0; j < impulseCnt; ++j) {
    const int start = j * (signalLen / impulseCnt);
    const int frames = (signalLen * fillRatio) / (impulseCnt);
    for (int i = start; i < start + frames; ++i) {
      signal[i] = sin(static_cast<float>(i) * ratio) * maxAmplitude;
    }
  }

  augmentation::Reverberation::Config config;
  // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
  config.debugLevel_ = 3;
  config.debugOutputPath_ = "/tmp";
  config.lengthMilliseconds_ = 1;

  std::vector<float> augmented;
  for (int m = 1; m < 1000; m *= 10) {
    config.distanceToObjectInMetersMin_ = m;
    config.distanceToObjectInMetersMax_ = m;
    for (int e = 1; e < 100; e *= 4) {
      config.echoCount_ = e;
      for (float a = 0.001; a < 1; a *= 4) {
        config.absorptionCoefficientMin_ = a;
        config.absorptionCoefficientMax_ = a;

        std::stringstream filename;
        filename << "reverb-echos-" << e << "-dist-" << m << "m-absorb-" << a;
        config.debugOutputFilePrefix_ = filename.str();

        augmentation::Reverberation reveberation(config);
        reveberation.enable(true);

        augmented = signal;
        reveberation.augment(&augmented);
        EXPECT_EQ(augmented.size(), signal.size());
      }
    }

    std::cout << "augmented-signal={";
    for (int i = sampleRate; i < sampleRate * 1.01; ++i) {
      std::cout << augmented[i] - signal[i] << ", ";
    }
    std::cout << "}" << std::endl;
  }
}

// TEST(Reverberation, LogAugmentedSamplesWithVariousfConfigs) {
//   augmentation::AudioLoader signalDb(kSignalInputDirectory);
//   const std::vector<augmentation::AudioLoader::Audio> signals = {
//       signalDb.loadRandom(), signalDb.loadRandom(), signalDb.loadRandom()};

//   augmentation::Reverberation::Config config;
//   config.debugLevel_ =
//       3; // 0=none, 1=stats, 2=histogram, 3= saveq augmented files
//   config.debugOutputPath_ = "/tmp";
//   config.debugOutputFilePrefix_ = "/reverb-";
//   config.lengthMilliseconds_ = 1;

//   augmentation::Reverberation reveberation(config);
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
