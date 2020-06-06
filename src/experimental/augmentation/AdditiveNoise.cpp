// Copyright 2004-present Facebook. All Rights Reserved.

#include "experimental/augmentation/AdditiveNoise.h"

#include <algorithm>
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
  ss << "maxTimeRatio=" << maxTimeRatio << " minSnr=" << minSnr
     << " maxSnr=" << maxSnr << " nClipsPerUtterance=" << nClipsPerUtterance
     << " noiseDir=" << noiseDir << " debugLevel=" << debugLevel;
  return ss.str();
}

AdditiveNoise::AdditiveNoise(AdditiveNoise::Config config)
    : audioLoader_(config.noiseDir), config_(std::move(config)) {}

const size_t kHistogramBucketCount = 15;
const size_t kHistogramBucketMaxLen = 100;

void AdditiveNoise::augment(std::vector<float>& signal) {
  double noiseAvg = 0.0;
  AudioLoader::Audio noise;

  fl::HistogramStats<float> signalHist;
  if (config_.debugLevel > 1) {
    signalHist = fl::FixedBucketSizeHistogram<float>(
        signal.begin(), signal.end(), kHistogramBucketCount);
  }

  std::vector<AudioLoader::Audio> noises;
  for (int i = 0; i < config_.nClipsPerUtterance; ++i) {
    try {
      noises.push_back(audioLoader_.loadRandom());
    } catch (const std::exception& ex) {
      std::cerr << "AdditiveNoise::augment(signal) failed with error="
                << ex.what();
    }
  }

  AudioAndStats noiseAndStats =
      sumAudiosAndCalcStats(std::move(noises), signal.size());
  // Calculate signal stats. Move the signal data out of signal for caculation
  // and back in to signal before returning from this funciton.
  std::vector<float> tmpSignal;
  tmpSignal.swap(signal);
  AudioAndStats signalAndStats = calcAudioStats(std::move(tmpSignal));

  double snr = signalAndStats.absAvg_ / noiseAndStats.absAvg_;
  float noiseMultiplier = 1.0;
  if (snr < config_.minSnr) {
    noiseMultiplier = signalAndStats.absAvg_ / (config_.minSnr * noiseAvg);
  } else if (snr > config_.maxSnr) {
    noiseMultiplier = signalAndStats.absAvg_ / (config_.maxSnr * noiseAvg);
  }

  for (int i = 0; i < std::min(signal.size(), noiseAndStats.data_.size());
       ++i) {
    if (noiseAndStats.data_[i] != 0) {
      const float augmentedSample =
          signalAndStats.data_[i] + (noiseAndStats.data_[i] * noiseMultiplier);
      if (!std::isnan(augmentedSample)) {
        signalAndStats.data_[i] = augmentedSample;
      } else {
        std::cout << "AdditiveNoise::augment(signal) augmentedSample=NaN i="
                  << i << " signalAndStats.data_[i]=" << signalAndStats.data_[i]
                  << " noiseAndStats.data_[i]=" << noiseAndStats.data_[i]
                  << " noiseMultiplier"
                  << "  (noiseAndStats.data_[i] * noiseMultiplier)="
                  << (noiseAndStats.data_[i] * noiseMultiplier) << " snr=" << snr
                  << " config={" << config_.prettyString() << "}" << std::endl;
      }
    }
  }

  if (config_.debugLevel > 0) {
    static size_t idx = 0;
    ++idx;
    std::stringstream debug;
    debug << "AdditiveNoise::augment(signal) signalAndStats={" << signalAndStats.prettyString()
    << "} noiseAndStats={" << noiseAndStats.prettyString()
    << "} snr=" << snr << " noiseMultiplier=" << noiseMultiplier << " config={"
          << config_.prettyString() << "}";

    std::stringstream filename;
    if (config_.debugLevel > 2) {
      filename << "/tmp/augmented-with-" << noise.filename_ << "-idx-"
               << std::setfill('0') << std::setw(4) << idx << ".flac";
      debug << " saving augmented file=" << filename.str();
    }

    if (config_.debugLevel > 1) {
      fl::HistogramStats<float> noiseHist = fl::FixedBucketSizeHistogram<float>(
          noiseAndStats.data_.begin(),
          noiseAndStats.data_.end(),
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

    if (config_.debugLevel > 2) {
      saveSound(
          filename.str(),
          signal,
          noise.info_.samplerate,
          noise.info_.channels,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
  }
  signal.swap(signalAndStats.data_);
}

} // namespace augmentation
} // namespace w2l
