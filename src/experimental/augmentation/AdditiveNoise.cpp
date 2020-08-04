// Copyright 2004-present Facebook. All Rights Reserved.

#include "experimental/augmentation/AdditiveNoise.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>

#include "flashlight/common/Histogram.h"

namespace w2l {
namespace augmentation {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "maxTimeRatio_=" << maxTimeRatio_ << " minSnr_=" << minSnr_
     << " maxSnr_=" << maxSnr_ << " nClipsPerUtterance_=" << nClipsPerUtterance_
     << " noiseDir_=" << noiseDir_;
  return ss.str();
}

namespace {
std::vector<AudioLoader::Audio> loadNoises(
    size_t noiseCount,
    AudioLoader* audioLoader) {
  std::vector<AudioLoader::Audio> noiseAudioVec;
  for (int i = 0; i < noiseCount; ++i) {
    try {
      noiseAudioVec.push_back(audioLoader->loadRandom());
    } catch (const std::exception& ex) {
      std::cerr << "AdditiveNoise::apply(signal) failed with error="
                << ex.what();
    }
  }
  return noiseAudioVec;
}

// 1. Center the noise on randomCenter.
// 2. Trim noise to fit in [0..resultSize]
// 3. Mask with zeros the interval [augStart..augEnd]
// 4. Compute power of the result noise
std::pair<std::vector<float>, double> randomlyShiftAndCalcPower(
    const AudioLoader::Audio& noise,
    int resultSize,
    int randomCenter,
    int augStart,
    int augEnd) {
  std::vector<float> shiftedNoise(resultSize, 0.0);

  randomCenter %= resultSize;
  const int halfNoiseSize = noise.data_.size() / 2;
  const int srcStartIndex = std::max(halfNoiseSize - randomCenter, 0);
  const int dstStartIndex = std::max(randomCenter - halfNoiseSize, 0);
  const int dstEndIndex = std::min(randomCenter + halfNoiseSize, resultSize);

  int srcIndex = srcStartIndex;
  double sqrPower = 0;
  for (int dstIndex = dstStartIndex; dstIndex < dstEndIndex;
       ++dstIndex, ++srcIndex) {
    if (augStart >= dstIndex && dstIndex < augEnd) {
      continue;
    }
    const float sample = noise.data_.at(srcIndex);
    sqrPower += (sample * sample);
    shiftedNoise.at(dstIndex) += noise.data_.at(srcIndex);
  }
  const double power = sqrt(sqrPower);
  return {std::move(shiftedNoise), power};
}

} // namespace

void AdditiveNoise::reset() {
  uniformDistribution_.reset();
  audioLoader_.reset();
}

AdditiveNoise::AdditiveNoise(
    const SoundEffect::Config& sfxConfig,
    const AdditiveNoise::Config& noiseConfig)
    : SoundEffect(sfxConfig),
      randomEngine_(sfxConfig_.randomSeed_),
      uniformDistribution_(0, std::numeric_limits<int>::max()),
      noiseConfig_(noiseConfig),
      audioLoader_(noiseConfig.noiseDir_) {}

void AdditiveNoise::apply(
    std::vector<float>* signal,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  if (debugMsg) {
    *debugMsg << "AdditiveNoise::apply(signal->size()=" << signal->size() << ')'
              << std::endl;
  }
  if (debugFilename) {
    *debugFilename << "tracks-" << noiseConfig_.nClipsPerUtterance_
                   << std::setw(4) << std::setprecision(2) << std::setfill('0')
                   << "-ratio-" << noiseConfig_.maxTimeRatio_;
  }

  const int numAugSignalSamples =
      static_cast<double>(signal->size()) * noiseConfig_.maxTimeRatio_;
  if (numAugSignalSamples <= 0 || signal->empty()) {
    return;
  }

  // Calc noise augmentation boundarties using noiseConfig_.maxTimeRatio_
  const bool augAll = (signal->size() == numAugSignalSamples);
  const int augStart = augAll ? 0 : (uniformDistribution_(randomEngine_) %
                                     (signal->size() - numAugSignalSamples));
  const int augEnd = augStart + numAugSignalSamples;

  // Load noises
  const std::vector<AudioLoader::Audio> noiseAudioVec =
      loadNoises(noiseConfig_.nClipsPerUtterance_, &audioLoader_);

  // Combine normalized noises
  std::vector<float> noise(signal->size());
  for (int i = 0; i < noiseAudioVec.size(); ++i) {
    // center noise on random place of the signal
    const size_t randomNoiseCenter =
        uniformDistribution_(randomEngine_) % numAugSignalSamples;
    const std::pair<std::vector<float>, float> shiftedTrimmedNoiseAndPower =
        randomlyShiftAndCalcPower(
            noiseAudioVec[i],
            signal->size(),
            randomNoiseCenter,
            augStart,
            augEnd);

    if (shiftedTrimmedNoiseAndPower.second == 0) {
      if (debugMsg) {
        *debugMsg << "warning: noisePower=0" << std::endl;
      }
      continue;
    }

    for (int j = augStart; j < augEnd; ++j) {
      noise.at(j) += shiftedTrimmedNoiseAndPower.first.at(j) /
          shiftedTrimmedNoiseAndPower.second;
    }
  }

  // Calc signal and noise power
  double noiseSqrPower = 0;
  double signalSqrPower = 0;
  for (int n = augStart; n < augEnd; ++n) {
    const float noiseSample = noise.at(n);
    const float signalsSample = signal->at(n);

    noiseSqrPower += (noiseSample * noiseSample);
    signalSqrPower += (signalsSample * signalsSample);
  }
  const double noisePower = sqrt(noiseSqrPower);
  const double signalPower = sqrt(signalSqrPower);

  // Augment signal with noise
  const double snr = signalPower / noisePower;
  double snrAdjustNoiseMultiplier = 1.0;
  if (snr < noiseConfig_.minSnr_) {
    snrAdjustNoiseMultiplier = signalPower / noiseConfig_.minSnr_;
  } else if (snr > noiseConfig_.maxSnr_) {
    snrAdjustNoiseMultiplier = signalPower / noiseConfig_.maxSnr_;
  }

  // Augment signal
  for (int j = augStart; j < augEnd; ++j) {
    signal->at(j) += noise.at(j) * snrAdjustNoiseMultiplier;
  }

  if (debugFilename) {
    *debugFilename << "-snr-" << std::setw(6) << std::setprecision(4)
                   << std::setfill('0') << (snr / snrAdjustNoiseMultiplier);
  }
}

std::string AdditiveNoise::prettyString() const {
  std::stringstream ss;
  ss << "audioLoader_=" << audioLoader_.prettyString()
     << " config=" << noiseConfig_.prettyString() << " SoundEffect={"
     << SoundEffect::prettyString() << "}}";
  return ss.str();
};

} // namespace augmentation
} // namespace w2l
