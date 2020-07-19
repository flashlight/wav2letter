// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <random>
#include <string>
#include <vector>

#include "experimental/augmentation/AudioLoader.h"
#include "experimental/augmentation/SoundEffect.h"

namespace w2l {
namespace augmentation {

class Reverberation : public SoundEffect {
 public:
  struct Config {
    std::string impulseResponseDir_;
    int lengthMilliseconds_ = 0;

    // https://www.acoustic-supplies.com/absorption-coefficient-chart/
    float absorptionCoefficientMin_ = 0.01; // painted brick
    float absorptionCoefficientMax_ = 0.99; // best absorptive wall materials
    float distanceToWallInMetersMin_ = 1.0;
    float distanceToWallInMetersMax_ = 50.0;
    size_t numWallsMin_ = 1; // number of pairs of refelctive objects
    size_t numWallsMax_ = 4; // number of pairs of refelctive objects
    float jitter_ = 0.1;
    size_t sampleRate_ = 16000;

    std::string prettyString() const;
  };

   Reverberation(
      const SoundEffect::Config& sfxConfig,
      const Reverberation::Config& reverbConfig);
  ~Reverberation() override = default;

  std::string prettyString() const override {
    return "Reverberation{config=" + reverbConfig_.prettyString() + "}";
  };

  std::string name() const override {
    return "Reverberation";
  };

 protected:
  void apply(
      std::vector<float>* signal,
      std::stringstream* debugMsg = nullptr,
      std::stringstream* debugFilename = nullptr) override;

 private:
  void conv1d(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugSaveAugmentedFileName);
  void randomShift(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugSaveAugmentedFileName);
  void randomShiftGab(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugSaveAugmentedFileName);
  void randomShiftGabCpu(
      const std::vector<float>& input,
      std::vector<float>* output,
      std::stringstream* debug,
      std::stringstream* debugSaveAugmentedFileName);

  // AudioLoader audioLoader_;
  const Reverberation::Config reverbConfig_;
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
