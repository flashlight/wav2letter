// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "experimental/augmentation/AudioLoader.h"
#include "experimental/augmentation/SoundEffect.h"

namespace w2l {
namespace augmentation {

class AdditiveNoise : public SoundEffect {
 public:
  struct Config {
    double maxTimeRatio_ = 1.0;
    double minSnr_ = 0.5;
    double maxSnr_ = 2.0;
    int nClipsPerUtterance_ = 1.0;
    std::string noiseDir_;

    std::string prettyString() const;
  };

  // class Random {
  //  public:
  //   explicit Random(unsigned int randomSeed);

  //   int index(int size);

  //  private:
  //   std::mt19937 randomEngine_;
  //   std::uniform_int_distribution<> uniformDistribution_;
  // };

  AdditiveNoise(
      const SoundEffect::Config& sfxConfig,
      const AdditiveNoise::Config& noiseConfig);
  ~AdditiveNoise() override = default;

  std::string prettyString() const override;

  std::string name() const override {
    return "AdditiveNoise";
  };

  void reset() override;

  void apply(
      std::vector<float>* signal,
      std::stringstream* debugMsg = nullptr,
      std::stringstream* debugFilename = nullptr) override;

 private:
  std::mt19937 randomEngine_;
  std::uniform_int_distribution<> uniformDistribution_;
  AudioLoader audioLoader_;
  AdditiveNoise::Config noiseConfig_;
  std::vector<std::string> noiseFilePathVec_;
  // Random random_;
};

} // namespace augmentation
} // namespace w2l
