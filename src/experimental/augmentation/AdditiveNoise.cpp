// Copyright 2004-present Facebook. All Rights Reserved.

#include "experimental/augmentation/AdditiveNoise.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include "flashlight/common/Histogram.h"

namespace w2l {
namespace augmentation {

std::string AdditiveNoise::Config::prettyString() const {
  std::stringstream ss;
  ss << "maxTimeRatio_=" << maxTimeRatio_ << " minSnr_=" << minSnr_
     << " maxSnr_=" << maxSnr_ << " nClipsPerUtterance_=" << nClipsPerUtterance_
     << " noiseDir_=" << noiseDir_ << " debugLevel_=" << debugLevel_;
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
      std::cerr << "AdditiveNoise::augment(signal) failed with error="
                << ex.what();
    }
  }
  return noises;
}

AudioLoader::Audio randomlyShiftAndSumNoise(
    const std::vector<AudioLoader::Audio>& noises,
    int len,
    std::mt19937* randomEngine) {
  std::uniform_int_distribution<> uniformDistribution(0, len);

  AudioLoader::Audio result;
  result.data_.resize(len);
  for (const AudioLoader::Audio& noise : noises) {
    const int cenerFrame = uniformDistribution(*randomEngine);
    const int halfNoiseLen = noise.data_.size() / 2;
    const int srcStartIndex = std::max(halfNoiseLen - cenerFrame, 0);
    const int dstStartIndex = std::min(cenerFrame - halfNoiseLen, 0);
    const int dstEndIndex = std::min(cenerFrame + halfNoiseLen, len);

    int srcIndex = srcStartIndex;
    for (int dstIndex = dstStartIndex; dstIndex < dstEndIndex; ++dstIndex) {
      result.data_[dstIndex] += noise.data_[srcIndex];
    }
  }
  return result;
}

float clampAudioFrameValue(float frameVal) {
  return std::min(std::max(frameVal, -1.0f), 1.0f);
}

void augmentNoiseToSignal(
    const std::vector<float> noise,
    double minSnr,
    double maxSnr,
    int debugLevel,
    std::vector<float>* signal) {
  assert(noise.size() == signal->size());
  assert(minSnr <= maxSnr);
  assert(minSnr > 0);

  AudioAndStats noiseAndStats(std::move(noise));
  AudioAndStats signalAndStats(std::move(*signal));

  double snr = signalAndStats.sqrAvg_ / noiseAndStats.sqrAvg_;
  float noiseMultiplier = 1.0;
  if (snr < minSnr) {
    noiseMultiplier = signalAndStats.sqrAvg_ / (minSnr * noiseAndStats.sqrAvg_);
  } else if (snr > maxSnr) {
    noiseMultiplier = signalAndStats.sqrAvg_ / (maxSnr * noiseAndStats.sqrAvg_);
  }

  for (int i = 0; i < signalAndStats.data_.size(); ++i) {
    if (noiseAndStats.data_[i] != 0) {
      signalAndStats.data_[i] = clampAudioFrameValue(
          signalAndStats.data_[i] + noiseAndStats.data_[i] * noiseMultiplier);
    }
  }

  if (debugLevel > 1) {
    std::stringstream debug;
    debug << "augmentNoiseToSignal(noise={" << noiseAndStats.prettyString()
          << " minSnr=" << minSnr << " maxSnr=" << maxSnr
          << " debugLevel=" << debugLevel << " signal={"
          << signalAndStats.prettyString() << "}) signalAndStats={"
          << signalAndStats.prettyString() << "})  snr=" << snr
          << " noiseMultiplier=" << noiseMultiplier;
    std::cout << debug.str() << std::endl;
  }

  signal->swap(signalAndStats.data_);
}

} // namespace

AdditiveNoise::AdditiveNoise(AdditiveNoise::Config config)
    : audioLoader_(config.noiseDir_),
      config_(std::move(config)),
      randomEngine_(config_.randomSeed_) {}

const size_t kHistogramBucketCount = 15;
const size_t kHistogramBucketMaxLen = 100;

void AdditiveNoise::augment(std::vector<float>& signal) {
  double noiseAvg = 0.0;

  fl::HistogramStats<float> signalHist;
  std::stringstream debug;
  if (config_.debugLevel_ > 1) {
    signalHist = fl::FixedBucketSizeHistogram<float>(
        signal.begin(), signal.end(), kHistogramBucketCount);

    debug << "AdditiveNoise::augment(signal.size()=" << signal.size()
          << ") config={" << config_.prettyString() << "}";
  }

  std::vector<AudioLoader::Audio> noises =
      loadNoises(config_.nClipsPerUtterance_, &audioLoader_);

  AudioLoader::Audio randomlyShiftedNoiseSum =
      randomlyShiftAndSumNoise(noises, signal.size(), &randomEngine_);

  augmentNoiseToSignal(
      randomlyShiftedNoiseSum.data_,
      config_.minSnr_,
      config_.maxSnr_,
      config_.debugLevel_,
      &signal);

  if (config_.debugLevel_ > 0) {
    std::stringstream filename;
    if (config_.debugLevel_ > 2) {
      static size_t idx = 0;
      ++idx;

      filename << "/tmp/augmented-with-";
      for (const AudioLoader::Audio& noise : noises) {
        filename << noise.filename_ << '-';
      }
      filename << "-idx-" << std::setfill('0') << std::setw(4) << idx
               << ".flac";
      debug << " saving augmented file=" << filename.str();
    }

    if (config_.debugLevel_ > 1) {
      fl::HistogramStats<float> noiseHist = fl::FixedBucketSizeHistogram<float>(
          randomlyShiftedNoiseSum.data_.begin(),
          randomlyShiftedNoiseSum.data_.end(),
          kHistogramBucketCount);

      fl::HistogramStats<float> augmentedHist =
          fl::FixedBucketSizeHistogram<float>(
              signal.begin(), signal.end(), kHistogramBucketCount);

      debug << std::endl
            << "signal-Hist="
            << signalHist.prettyString(
                   kHistogramBucketMaxLen,
                   fl::shortFormatCount,
                   fl::shortFormatFloat<float>)
            << std::endl
            << "augmented-Hist="
            << augmentedHist.prettyString(
                   kHistogramBucketMaxLen,
                   fl::shortFormatCount,
                   fl::shortFormatFloat<float>)
            << std::endl
            << "noise-Hist="
            << noiseHist.prettyString(
                   kHistogramBucketMaxLen,
                   fl::shortFormatCount,
                   fl::shortFormatFloat<float>)
            << std::endl;
    }
    std::cout << debug.str() << std::endl;

    if (config_.debugLevel_ > 2) {
      saveSound(
          filename.str(),
          signal,
          noises.at(0).info_.samplerate,
          noises.at(0).info_.channels,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
  }
}

} // namespace augmentation
} // namespace w2l
