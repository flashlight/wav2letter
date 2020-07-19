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

#include "experimental/augmentation/AudioStats.h"
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
  std::vector<AudioLoader::Audio> noises;
  for (int i = 0; i < noiseCount; ++i) {
    try {
      noises.push_back(audioLoader->loadRandom());
    } catch (const std::exception& ex) {
      std::cerr << "AdditiveNoise::apply(signal) failed with error="
                << ex.what();
    }
  }
  return noises;
}

AudioLoader::Audio randomlyShiftAndSumNoise(
    const std::vector<AudioLoader::Audio>& noises,
    int len,
    AdditiveNoise::Random* random) {
  AudioLoader::Audio result;
  result.data_.resize(len, 0.0);
  for (const AudioLoader::Audio& noise : noises) {
    const int cenerFrame = random->index(len);
    const int halfNoiseLen = noise.data_.size() / 2;
    const int srcStartIndex = std::max(halfNoiseLen - cenerFrame, 0);
    const int dstStartIndex = std::max(cenerFrame - halfNoiseLen, 0);
    const int dstEndIndex = std::min(cenerFrame + halfNoiseLen, len);

    int srcIndex = srcStartIndex;
    for (int dstIndex = dstStartIndex; dstIndex < dstEndIndex;
         ++dstIndex, ++srcIndex) {
      result.data_.at(dstIndex) += noise.data_.at(srcIndex);
    }
  }
  return result;
}

void maskBeyondMaxTimeRatioRandomStart(
    double maxTimeRatio,
    std::vector<float>* data,
    AdditiveNoise::Random* random) {
  const int size = data->size();
  const int startFrame = random->index(size);
  const int numFrames = static_cast<double>(size) * (1.0 - maxTimeRatio);
  for (int i = startFrame; i < startFrame + numFrames; ++i) {
    data->at(i % size) = 0;
  }
}

void augmentNoiseToSignal(
    const std::vector<float>& noise,
    double minSnr,
    double maxSnr,
    std::vector<float>* signal,
    std::stringstream* debugMsg) {
  assert(noise.size() == signal->size());
  assert(minSnr <= maxSnr);
  assert(minSnr > 0);

  AudioStats noiseStats = calcAudioStats(noise);
  AudioStats signalStats = calcAudioStats(*signal);

  double snr = signalStats.sqrAvg_ / noiseStats.sqrAvg_;
  float noiseMultiplier = 1.0;
  if (snr < minSnr) {
    noiseMultiplier = signalStats.sqrAvg_ / (minSnr * noiseStats.sqrAvg_);
  } else if (snr > maxSnr) {
    noiseMultiplier = signalStats.sqrAvg_ / (maxSnr * noiseStats.sqrAvg_);
  }

  if (debugMsg) {
    *debugMsg << "augmentNoiseToSignal(noise={" << noiseStats.prettyString()
              << " minSnr=" << minSnr << " maxSnr=" << maxSnr
              << " signal->size()={" << signal->size() << "}) signalStats={"
              << signalStats.prettyString() << "})  snr=" << snr
              << " noiseMultiplier=" << noiseMultiplier << std::endl;
  }
}

} // namespace

AdditiveNoise::Random::Random(unsigned int randomSeed)
    : randomEngine_(randomSeed),
      uniformDistribution_(0, std::numeric_limits<int>::max()) {}

int AdditiveNoise::Random::index(int size) {
  return uniformDistribution_(randomEngine_) % size;
}

AdditiveNoise::AdditiveNoise(
    const SoundEffect::Config& sfxConfig,
    const AdditiveNoise::Config& noiseConfig)
    : SoundEffect(sfxConfig),
      randomEngine_(sfxConfig_.randomSeed_),
      uniformDistribution_(0, std::numeric_limits<int>::max()),
      noiseConfig_(noiseConfig),
      audioLoader_(noiseConfig.noiseDir_),
      random_(sfxConfig_.randomSeed_) {}

void AdditiveNoise::apply(
    std::vector<float>* signal,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  if (debugMsg) {
    *debugMsg << "AdditiveNoise::apply(signal->size()=" << signal->size()
              << ')';
  }

  const std::vector<AudioLoader::Audio> noises =
      loadNoises(noiseConfig_.nClipsPerUtterance_, &audioLoader_);

  AudioLoader::Audio randomlyShiftedNoiseSum =
      randomlyShiftAndSumNoise(noises, signal->size(), &random_);

  maskBeyondMaxTimeRatioRandomStart(
      noiseConfig_.maxTimeRatio_, &randomlyShiftedNoiseSum.data_, &random_);

  std::stringstream augmentNoiseToSignalDebugMsg;
  augmentNoiseToSignal(
      randomlyShiftedNoiseSum.data_,
      noiseConfig_.minSnr_,
      noiseConfig_.maxSnr_,
      signal,
      debugMsg);

  if (debugFilename) {
    *debugFilename << "-tracks-" << noises.size() << "-snr-" << std::setw(6)
                   << std::setprecision(4) << std::setfill('0')
                   << noiseConfig_.minSnr_ << "-" << std::setw(6)
                   << std::setprecision(4) << std::setfill('0')
                   << noiseConfig_.maxSnr_ << "-ratio-" << std::setw(4)
                   << std::setprecision(2) << std::setfill('0')
                   << noiseConfig_.maxTimeRatio_;
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
