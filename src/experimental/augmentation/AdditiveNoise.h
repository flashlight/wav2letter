// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "experimental/augmentation/AudioAugmenter.h"
#include "experimental/augmentation/AudioLoader.h"

namespace w2l {
namespace augmentation {

class AdditiveNoise : public SoundEffect {
 public:
  struct Config {
    unsigned int randomSeed_ = std::mt19937::default_seed;
    double maxTimeRatio_;
    double minSnr_;
    double maxSnr_;
    int nClipsPerUtterance_;
    std::string noiseDir_;
    int debugLevel_; // 0=none, 1=stats, 2=histogram, 3=save augmented files
    std::string debugOutputPath_ = "/tmp";
    std::string debugOutputFilePrefix_ = "/additive-noise-";

    std::string prettyString() const;
  };

  class Random {
   public:
    explicit Random(unsigned int randomSeed);

    int index(int size);

   private:
    std::mt19937 randomEngine_;
    std::uniform_int_distribution<> uniformDistribution_;
  };

  explicit AdditiveNoise(AdditiveNoise::Config config);
  ~AdditiveNoise() override = default;

  void apply(std::vector<float>* signal, std::stringstream* debugSaveAugmentedFileName) override;

  std::string prettyString() const override {
    return "AdditiveNoise{config=" + config_.prettyString() + "}";
  };

  std::string name() const override {
    return "AdditiveNoise";
  };

 private:
  int randomIndexGenerator(int size);

  AudioLoader audioLoader_;
  const AdditiveNoise::Config config_;
  std::vector<std::string> noiseFilePathVec_;
  Random random_;
};

} // namespace augmentation
} // namespace w2l
