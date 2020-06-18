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

class AdditiveNoise : public AudioAugmenter {
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

  AdditiveNoise(AdditiveNoise::Config config);

  void augment(std::vector<float>* signal) override;

 private:
  AudioLoader audioLoader_;
  const AdditiveNoise::Config config_;
  std::vector<std::string> noiseFilePathVec_;
  std::mt19937 randomEngine_;
};

} // namespace augmentation
} // namespace w2l
