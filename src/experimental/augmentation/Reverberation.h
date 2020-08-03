// Copyright 2004-present Facebook. All Rights Reserved.

#pragma once

#include <memory>
#include <mutex>
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
    int lengthMilliseconds_ = 900;

    // https://www.acoustic-supplies.com/absorption-coefficient-chart/
    float absorptionCoefficientMin_ = 0.01; // painted brick
    float absorptionCoefficientMax_ = 0.99; // best absorptive wall materials
    float distanceToWallInMetersMin_ = 1.0;
    float distanceToWallInMetersMax_ = 50.0;
    size_t numWallsMin_ = 1; // number of pairs of refelctive objects
    size_t numWallsMax_ = 4; // number of pairs of refelctive objects
    float jitter_ = 0.1;
    size_t sampleRate_ = 16000;
    enum Backend {
      GPU_GAB = 0,
      CPU_GAB = 1,
      GPU_CONV = 2,
    };
    static Config::Backend backendFromString(const std::string& backend) {
      if (backend == "gpu_gab") {
        return Backend::GPU_GAB;
      } else if (backend == "cpu_gab") {
        return Backend::CPU_GAB;
      } else {
        return Backend::GPU_CONV;
      }
    }
    static std::string backendPrettyString(Config::Backend backend) {
      switch (backend) {
        case Backend::GPU_GAB:
          return "gpu_gab";
        case Backend::CPU_GAB:
          return "cpu_gab";
        case Backend::GPU_CONV:
          return "gpu_conv";
        default:
          return std::string("no_such_backend=") +
              std::to_string(static_cast<int>(backend));
      }
    };
    Backend backend_ = GPU_GAB;

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

  void apply(
      std::vector<float>* signal,
      std::stringstream* debugMsg = nullptr,
      std::stringstream* debugFilename = nullptr) override;

 private:
  void conv1d(
      const std::vector<float>& input,
      std::vector<float>* output,
      float firstDelay,
      float rt60,
      int numWalls);
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
      float firstDelay,
      float rt60,
      int numWalls);
  void randomShiftGabGpu(
      const std::vector<float>& input,
      std::vector<float>* output,
      float firstDelay,
      float rt60,
      int numWalls);

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

  std::mutex mtx_;
  int memUsed_ = 0;
  int maxMemUsed_ = 0;
  int maxMemUsedForOne_ = 0;
};

} // namespace augmentation
} // namespace w2l
