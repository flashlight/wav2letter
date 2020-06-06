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
    double maxTimeRatio;
    double minSnr;
    double maxSnr;
    int nClipsPerUtterance;
    std::string noiseDir;
    int debugLevel;  // 0=none, 1=stats, 2=histogram, 3=save augmented files

    std::string prettyString() const;
  };

  AdditiveNoise(AdditiveNoise::Config config);

  void augment(std::vector<float>& signal) override;

 private:
  AudioLoader audioLoader_;
  const AdditiveNoise::Config config_;
  std::vector<std::string> noiseFilePathVec_;
};

} // namespace augmentation
} // namespace w2l
