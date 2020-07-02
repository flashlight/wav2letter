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
      std::cerr << "AdditiveNoise::augmentImpl(signal) failed with error="
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

float clampAudioFrameValue(float frameVal) {
  return std::min(std::max(frameVal, -1.0f), 1.0f);
}

void augmentNoiseToSignal(
    const std::vector<float>& noise,
    double minSnr,
    double maxSnr,
    std::vector<float>* signal,
    std::stringstream* debug) {
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

  for (int i = 0; i < signal->size(); ++i) {
    if (noise[i] != 0) {
      (*signal)[i] =
          clampAudioFrameValue((*signal)[i] + noise[i] * noiseMultiplier);
    }
  }

  if (debug) {
    (*debug) << "augmentNoiseToSignal(noise={" << noiseStats.prettyString()
             << " minSnr=" << minSnr << " maxSnr=" << maxSnr
             << " signal->size()={" << signal->size() << "}) signalStats={"
             << signalStats.prettyString() << "})  snr=" << snr
             << " noiseMultiplier=" << noiseMultiplier;
  }
}

} // namespace

AdditiveNoise::Random::Random(unsigned int randomSeed)
    : randomEngine_(randomSeed),
      uniformDistribution_(0, std::numeric_limits<int>::max()) {
  // std::srand(randomSeed);
}

int AdditiveNoise::Random::index(int size) {
  // return std::rand() % size;
  return uniformDistribution_(randomEngine_) % size;
}

AdditiveNoise::AdditiveNoise(AdditiveNoise::Config config)
    : audioLoader_(config.noiseDir_),
      config_(std::move(config)),
      random_(config_.randomSeed_) {}

const size_t kHistogramBucketCount = 15;
const size_t kHistogramBucketMaxLen = 100;

void AdditiveNoise::augmentImpl(std::vector<float>* signal) {
  fl::HistogramStats<float> signalHist;
  std::stringstream debug;
  if (config_.debugLevel_ > 1) {
    signalHist = fl::FixedBucketSizeHistogram<float>(
        signal->begin(), signal->end(), kHistogramBucketCount);

    debug << "AdditiveNoise::augmentImpl(signal->size()=" << signal->size()
          << ") config={" << config_.prettyString() << "}";
  }

  const std::vector<AudioLoader::Audio> noises =
      loadNoises(config_.nClipsPerUtterance_, &audioLoader_);

  AudioLoader::Audio randomlyShiftedNoiseSum =
      randomlyShiftAndSumNoise(noises, signal->size(), &random_);

  maskBeyondMaxTimeRatioRandomStart(
      config_.maxTimeRatio_, &randomlyShiftedNoiseSum.data_, &random_);

  std::stringstream augmentNoiseToSignalDebugMsg;
  augmentNoiseToSignal(
      randomlyShiftedNoiseSum.data_,
      config_.minSnr_,
      config_.maxSnr_,
      signal,
      (config_.debugLevel_ ? &augmentNoiseToSignalDebugMsg : nullptr));

  if (config_.debugLevel_ > 0) {
    debug << augmentNoiseToSignalDebugMsg.str() << std::endl;

    std::stringstream filename;
    static size_t idx = 0;
    ++idx;
    filename << config_.debugOutputPath_ << config_.debugOutputFilePrefix_
             << "-augmenting-" << noises.size() << "-noises-snr-"
             << std::setw(6) << std::setprecision(4) << std::setfill('0')
             << config_.minSnr_ << "-" << std::setw(6) << std::setprecision(4)
             << std::setfill('0') << config_.maxSnr_ << "-ratio-"
             << std::setw(4) << std::setprecision(2) << std::setfill('0')
             << config_.maxTimeRatio_ << "-idx-" << std::setfill('0')
             << std::setw(4) << idx;
    std::string metaDataFilename = filename.str() + ".txt";
    std::string audioFilename = filename.str() + ".flac";

    for (const AudioLoader::Audio& noise : noises) {
      debug << "augmenting noise file=" << noise.filename_ << std::endl;
    }
    if (config_.debugLevel_ > 2) {
      debug << "saving augmented file=" << audioFilename;
    }

    // Add histograms
    if (config_.debugLevel_ > 1) {
      fl::HistogramStats<float> noiseHist = fl::FixedBucketSizeHistogram<float>(
          randomlyShiftedNoiseSum.data_.begin(),
          randomlyShiftedNoiseSum.data_.end(),
          kHistogramBucketCount);

      fl::HistogramStats<float> augmentedHist =
          fl::FixedBucketSizeHistogram<float>(
              signal->begin(), signal->end(), kHistogramBucketCount);

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
      std::ofstream infoFile(metaDataFilename);
      if (infoFile.is_open()) {
        infoFile << debug.str() << std::endl;
      }

      saveSound(
          audioFilename,
          *signal,
          noises.at(0).info_.samplerate,
          noises.at(0).info_.channels,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
  }
}

} // namespace augmentation
} // namespace w2l
