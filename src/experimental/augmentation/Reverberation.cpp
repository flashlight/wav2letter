// Copyright 2004-present Facebook. All Rights Reserved.

#include "experimental/augmentation/Reverberation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

#include <arrayfire.h>

#include "experimental/augmentation/AudioStats.h"
#include "flashlight/autograd/Functions.h"
#include "flashlight/autograd/Variable.h"
#include "flashlight/common/Histogram.h"
#include "flashlight/nn/Init.h"
#include "flashlight/nn/modules/Conv2D.h"

namespace w2l {
namespace augmentation {

// https://en.wikipedia.org/wiki/Reverberation#Sabine_equation
// Asuuming
// distance to the 4 walls is the same and room hight is 2 meters.
// V =~ (distanceMeters*2)^2 * 2
// A =~ distanceMeters^2 * 2
// V/A = 4
float rt60(float absorptionCoefficien) {
  return (0.161 * 4) / absorptionCoefficien;
}

Reverberation::Reverberation(Reverberation::Config config)
    : config_(std::move(config)),
      randomEngine_(config_.randomSeed_),
      random_impulse_(-1.0, 1.0),
      randomDecay_(
          1.0 / rt60(config_.absorptionCoefficientMin_),
          1.0 / rt60(config_.absorptionCoefficientMax_)),
      randomDelay_(
          config_.distanceToObjectInMetersMin_ / kSpeedOfSoundMeterPerSec,
          config_.distanceToObjectInMetersMax_ / kSpeedOfSoundMeterPerSec) {
  std::cout << "Reverberation::Reverberation(config={" << config_.prettyString()
            << "})" << std::endl;
}

std::string Reverberation::Config::prettyString() const {
  std::stringstream ss;
  ss << " absorptionCoefficientMin_=" << absorptionCoefficientMin_
     << " absorptionCoefficientMax_=" << absorptionCoefficientMax_
     << " distanceToObjectInMetersMin_=" << distanceToObjectInMetersMin_
     << " distanceToObjectInMetersMax_=" << distanceToObjectInMetersMax_
     << " echoCount_=" << echoCount_ << " jitter_=" << jitter_
     << " sampleRate_=" << sampleRate_
     << " impulseResponseDir_=" << impulseResponseDir_
     << " lengthMilliseconds_=" << lengthMilliseconds_
     << " debugLevel_=" << debugLevel_
     << " debugOutputPath_=" << debugOutputPath_
     << " debugOutputFilePrefix_=" << debugOutputFilePrefix_
     << " randomSeed_=" << randomSeed_;
  return ss.str();
}

namespace {

fl::Variable randomShiftEcho(
    const fl::Variable source,
    const Reverberation::Config& config,
    const float firstDelay,
    float rt60,
    std::mt19937* randomEngine,
    std::uniform_real_distribution<float>* randomUnit,
    std::stringstream* debug) {
  fl::Variable reverb(source.array().copy(), false);
  for (int i = 0; i < config.echoCount_; ++i) {
    float frac = 1.0;
    fl::Variable echo = source * config.absorptionCoefficientMin_;
    size_t sumDelay = 0;
    while (frac > 1e-3 && sumDelay < source.elements()) {
      const float jitter = 1 + config.jitter_ * (*randomUnit)(*randomEngine);
      size_t delay = std::min(
          1UL + static_cast<size_t>(jitter * firstDelay * config.sampleRate_),
          static_cast<size_t>(source.elements()));
      sumDelay += delay;

      // Delay the echo in time by padding with zero on the left
      echo = fl::padding(
          echo, std::vector<std::pair<int, int>>({{delay, 0}}), /*val=*/0);

      // trim echo to length of reverb
      af::array& echoArray = echo.array();
      echoArray = echoArray(af::seq(1, reverb.elements()));

      reverb = reverb + echo;

      const float attenuationRandomness =
          1 + config.jitter_ * (*randomUnit)(*randomEngine);
      const float attenuation =
          pow(10, -3 * attenuationRandomness * firstDelay / rt60);
      frac *= attenuation;
      if (debug) {
        // *debug << "{delay=" << delay << "("
        //        << (delay / (config.sampleRate_ / 1000))
        //        << "ms) attenuation=" << attenuation << " frac=" << frac
        //        << "}, ";

        // std::cout << "{delay=" << delay << "("
        //           << (delay / (config.sampleRate_ / 1000))
        //           << "ms) attenuation=" << attenuation << " frac=" << frac
        //           << " firstDelay=" << firstDelay << "("
        //           << (firstDelay / (config.sampleRate_ / 1000)) << "ms)"
        //           << " rt60=" << rt60
        //           << " =firstDelay/rt60=" << firstDelay / rt60
        //           << " sumDelay=" << sumDelay << "("
        //           << (sumDelay * kSpeedOfSoundMeterPerSec) << "m)"
        //           << "}, " << std::endl;
      }
      echo = echo * attenuation;
    }
    if (debug) {
      *debug << std::endl;
    }
  }

  return reverb;
}
} // namespace

void Reverberation::randomShift(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debug) {
  const float firstDelay = randomDelay_(randomEngine_);
  const float rt60 = randomDecay_(randomEngine_);
  if (debug) {
    *debug << "firstDelay=" << firstDelay << "("
           << (firstDelay * kSpeedOfSoundMeterPerSec) << "m)"
           << " rt60=" << rt60 << std::endl;
  }

  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable inputAsVariable(inputAsAfArray, false);
  fl::Variable augmented = randomShiftEcho(
      inputAsVariable,
      config_,
      firstDelay,
      rt60,
      &randomEngine_,
      &random_impulse_,
      debug);

  augmented.host(output->data());
}

namespace {

void generateImpulseResponse(
    float* impulseResponseFrames,
    size_t impulseResponseFramesCount,
    std::mt19937* randomEngine,
    std::uniform_real_distribution<float>* random_impulse) {
  const float amplitudeDiffPerFrame =
      0.5f / static_cast<float>(impulseResponseFramesCount);
  for (int i = 0; i < impulseResponseFramesCount; ++i) {
    const float decrasingAmpltitude = amplitudeDiffPerFrame * i;
    const float randomImpulse = (*random_impulse)(*randomEngine);
    impulseResponseFrames[i] = decrasingAmpltitude * randomImpulse;
  }
}

fl::Variable generateImpulseResponseKernel(
    size_t len,
    std::mt19937* randomEngine,
    std::uniform_real_distribution<float>* random_impulse,
    int debugLevel,
    std::stringstream* debug) {
  const size_t kernelsize = len * 2;
  std::vector<float> kernelVector(kernelsize, 0);
  // Add inpulse response to the first half of the kernel.
  generateImpulseResponse(
      kernelVector.data(), len, randomEngine, random_impulse);

  af::array kernelArray(kernelVector.size(), 1, 1, 1, kernelVector.data());
  return fl::Variable(kernelArray, false);
}

} // namespace

const size_t kSampleRate = 16000;

void Reverberation::conv1d(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debug) {
  fl::Variable kernel = generateImpulseResponseKernel(
      config_.lengthMilliseconds_ * (kSampleRate / 1000),
      &randomEngine_,
      &random_impulse_,
      config_.debugLevel_,
      debug);

  fl::Conv2D reverbConv(kernel, 1, 1, kernel.elements() / 2);
  *debug << "reverbConv=" << reverbConv.prettyString() << std::endl;

  af::array signalArray(input.size(), input.data());
  fl::Variable signalVariable(signalArray, false);

  fl::Variable augmentedVariable = reverbConv.forward(signalVariable);
  augmentedVariable.host(output->data());
}

void Reverberation::augmentImpl(std::vector<float>* signal) {
  std::stringstream debug;
  std::string audioFilename;
  std::string metaDataFilename;

  if (config_.debugLevel_ > 1) {
    debug << "Reverberation::augmentImpl(signal->size()=" << signal->size()
          << ") config={" << config_.prettyString() << "}" << std::endl;
    static size_t idx = 0;
    ++idx;
    std::stringstream filename;
    filename << config_.debugOutputPath_ << '/'
             << config_.debugOutputFilePrefix_ << "-idx-" << std::setfill('0')
             << std::setw(4) << idx;
    metaDataFilename = filename.str() + ".txt";
    audioFilename = filename.str() + ".flac";
    std::string dryAudioFilename = filename.str() + "-dry.flac";

    if (config_.debugLevel_ > 2) {
      debug << "saving dry file=" << dryAudioFilename << std::endl;
      debug << "saving augmented file=" << audioFilename << std::endl;
      saveSound(
          dryAudioFilename,
          *signal,
          16000,
          1,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
    // std::cout << debug.str() << std::endl;
  }

  std::vector<float> augmented(signal->size(), 0);
  // conv1d(*signal, &augmented, &debug);
  randomShift(*signal, &augmented, &debug);

  if (config_.debugLevel_ > 1) {
    std::cout << debug.str() << std::endl;
    if (config_.debugLevel_ > 2) {
      std::ofstream infoFile(metaDataFilename);
      if (infoFile.is_open()) {
        infoFile << debug.str() << std::endl;
      }

      saveSound(
          audioFilename,
          augmented,
          16000,
          1,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
  }
  signal->swap(augmented);
}

} // namespace augmentation
} // namespace w2l
