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
    int debugLevel_; // 0=none, 1=stats, 2=histogram, 3=save augmented files, 4=save signal
    std::string debugOutputPath_ = "/tmp";
    std::string debugOutputFilePrefix_ = "reverb-";

  // https://www.acoustic-supplies.com/absorption-coefficient-chart/
    float absorptionCoefficientMin_ = 0.01; // painted brick
    float absorptionCoefficientMax_ = 0.99; // best absorptive wall materials
    float distanceToWallInMetersMin_ = 1.0;
    float distanceToWallInMetersMax_ = 50.0;
    size_t numWallsMin_ = 4;
    size_t numWallsMax_ = 16;
    float jitter_ = 0.1;
    size_t sampleRate_ = 16000;

    std::string prettyString() const;
  };

  explicit Reverberation(Reverberation::Config config);

  void augmentImpl(std::vector<float>* signal) override;

  std::string prettyString() const override {
    return "Reverberation{config=" + config_.prettyString() + "}";
  };

 private:
  void conv1d(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugFileName);
  void randomShift(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugFileName);
  void randomShiftGab(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugFileName);

  // AudioLoader audioLoader_;
  const Reverberation::Config config_;
  std::vector<std::string> impulseResponseFilePathVec_;
  std::mt19937 randomEngine_;
  std::uniform_real_distribution<float> randomZeroToOne_;
  std::uniform_real_distribution<float> randomUnit_;
  std::uniform_real_distribution<float> randomDecay_;
  std::uniform_real_distribution<float> randomDelay_;
  std::uniform_real_distribution<float> randomAbsorptionCoefficient_;
  std::uniform_real_distribution<float> randomDistanceToWallInMeters_;
  std::uniform_int_distribution<int> randomNumWalls_;
};

} // namespace augmentation
} // namespace w2l
