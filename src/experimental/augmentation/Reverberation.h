// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "experimental/augmentation/AudioAugmenter.h"
#include "experimental/augmentation/AudioLoader.h"

namespace w2l {
namespace augmentation {

constexpr float kSpeedOfSoundMeterPerSec = 343.0;

class Reverberation : public AudioAugmenter {
 public:
  struct Config {
    unsigned int randomSeed_ = std::mt19937::default_seed;
    std::string impulseResponseDir_;
    int lengthMilliseconds_ = 0;
    int debugLevel_; // 0=none, 1=stats, 2=histogram, 3=save augmented files
    std::string debugOutputPath_ = "/tmp";
    std::string debugOutputFilePrefix_ = "/reverb-";

    // float initialAplitudeMultiplier_ = 0.3; // initial
    // std::pair<float, float> decayRangeSec_ = {0.3, 1.3}; // rt60
    // std::pair<float, float> delayRangeSec_ = {0.01, 0.03};

    float absorptionCoefficientMin_ = 3.3;
    float absorptionCoefficientMax_ = 0.77;
    float distanceToObjectInMetersMin_ = 2.0;
    float distanceToObjectInMetersMax_ = 10.0;
    size_t echoCount_ = 3;
    float jitter_ = 0.1;
    size_t sampleRate_ = 16000;

    std::string prettyString() const;
  };

  Reverberation(Reverberation::Config config);

  void augmentImpl(std::vector<float>* signal) override;

  std::string prettyString() const override {
    return "Reverberation{config=" + config_.prettyString() + "}";
  };

 private:
  void conv1d(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug);
  void randomShift(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug);

  // AudioLoader audioLoader_;
  const Reverberation::Config config_;
  std::vector<std::string> impulseResponseFilePathVec_;
  std::mt19937 randomEngine_;
  std::uniform_real_distribution<float> random_impulse_;
  std::uniform_real_distribution<float> randomDecay_;
  std::uniform_real_distribution<float> randomDelay_;
};

} // namespace augmentation
} // namespace w2l
