// Copyright 2004-present Facebook. All Rights Reserved.

#include "experimental/augmentation/Reverberation.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
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

constexpr float kDistToAreasRatio = 100.0;
constexpr float kSabineConstant = 0.1611;
// Reveberation time
// https://en.wikipedia.org/wiki/Reverberation#Sabine_equation
// t60 = 0.1611*(V/(S*a))
// Asuuming
// distance to the 4 walls is the same and room hight is 2 meters.
// Volume proportinal to distanceMeters^3
// if we are
//  V =~ (distanceMeters*2)^3
// Surface area is proportinal to distanceMeters^2
//  A =~ (distanceMeters*2)^2 * 2
// V/A = distanceMeters/
// a is the abosorption coefficient
float calcRt60(float distanceMeters, float absorptionCoefficient) {
  return (kSabineConstant * distanceMeters) /
      (absorptionCoefficient * kDistToAreasRatio);
}

float absorptionCoefficient(float distanceMeters, float rt60) {
  return (kSabineConstant * distanceMeters) / (rt60 * kDistToAreasRatio);
}

Reverberation::Reverberation(Reverberation::Config config)
    : config_(std::move(config)),
      randomEngine_(config_.randomSeed_),
      randomZeroToOne_(0.0, 1.0),
      randomUnit_(-1.0, 1.0),
      randomDecay_(
          calcRt60(
              config_.distanceToWallInMetersMax_,
              config_.absorptionCoefficientMin_),
          calcRt60(
              config_.distanceToWallInMetersMin_,
              config_.absorptionCoefficientMax_)),
      randomDelay_(
          config_.distanceToWallInMetersMin_ / kSpeedOfSoundMeterPerSec,
          config_.distanceToWallInMetersMax_ / kSpeedOfSoundMeterPerSec),
      randomAbsorptionCoefficient_(
          config_.absorptionCoefficientMin_,
          config_.absorptionCoefficientMax_),
      randomDistanceToWallInMeters_(
          config_.distanceToWallInMetersMax_,
          config_.distanceToWallInMetersMin_),
      randomNumWalls_(
          static_cast<int>(config_.numWallsMin_),
          static_cast<int>(config_.numWallsMax_)) {
  assert(config_.numWallsMin_ <= config_.numWallsMax_);
  std::cout << "Reverberation::Reverberation(config={" << config_.prettyString()
            << "})" << std::endl;
}

std::string Reverberation::Config::prettyString() const {
  std::stringstream ss;
  ss << " absorptionCoefficientMin_=" << absorptionCoefficientMin_
     << " absorptionCoefficientMax_=" << absorptionCoefficientMax_
     << " distanceToWallInMetersMin_=" << distanceToWallInMetersMin_
     << " distanceToWallInMetersMax_=" << distanceToWallInMetersMax_
     << " numWallsMin_=" << numWallsMin_ << " numWallsMax_=" << numWallsMax_
     << " jitter_=" << jitter_ << " sampleRate_=" << sampleRate_
     << " impulseResponseDir_=" << impulseResponseDir_
     << " lengthMilliseconds_=" << lengthMilliseconds_
     << " debugLevel_=" << debugLevel_
     << " debugOutputPath_=" << debugOutputPath_
     << " debugOutputFilePrefix_=" << debugOutputFilePrefix_
     << " randomSeed_=" << randomSeed_;
  return ss.str();
}

namespace {

fl::Variable randomShiftGabEchos(
    const fl::Variable source,
    const Reverberation::Config& config,
    const float firstDelay,
    float rt60,
    int numWalls,
    std::mt19937* randomEngine,
    std::uniform_real_distribution<float>* randomUnit,
    std::stringstream* debug) {
  fl::Variable reverb(source.array().copy(), false);
  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
    fl::Variable echo(source.array().copy(), false);
    echo = echo / numWalls;
    size_t sumDelay = 0;

    while (frac > 1e-3 && sumDelay < source.elements()) {
      const float jitter = 1 + config.jitter_ * (*randomUnit)(*randomEngine);
      size_t delay = std::min(
          1UL + static_cast<size_t>(jitter * firstDelay * config.sampleRate_),
          static_cast<size_t>(source.elements()));
      sumDelay += delay;

      const float attenuationRandomness =
          1.0f + config.jitter_ * (*randomUnit)(*randomEngine);
      const float attenuation =
          pow(10, -3 * attenuationRandomness * firstDelay / rt60);
      frac *= attenuation;
      echo = echo * attenuation;

      // Delay the echo in time by padding with zero on the left
      echo = fl::padding(
          echo, std::vector<std::pair<int, int>>({{delay, 0}}), /*val=*/0);

      // trim echo to length of reverb
      af::array& echoArray = echo.array();
      echoArray = echoArray(af::seq(1, reverb.elements()));

      reverb = reverb + echo;
    }
  }

  return reverb;
}

} // namespace

void Reverberation::randomShiftGab(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debug,
    std::stringstream* debugFileName) {
  const float firstDelay = randomDelay_(randomEngine_);
  const float rt60 = randomDecay_(randomEngine_);
  const int numWalls = randomNumWalls_(randomEngine_);
  if (debug) {
    *debug << "firstDelay=" << firstDelay << "("
           << (firstDelay * kSpeedOfSoundMeterPerSec) << "m)"
           << " rt60=" << rt60 << std::endl;
    if (debugFileName) {
      const float dist = (firstDelay * kSpeedOfSoundMeterPerSec);
      *debugFileName << "-dist-" << std::setprecision(2) << dist << "-absrb-"
                     << std::setprecision(3)
                     << absorptionCoefficient(dist, rt60) << "-echos-"
                     << numWalls << "-jitter-" << config_.jitter_;
    }
  }

  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable inputAsVariable(inputAsAfArray, false);
  fl::Variable augmented = randomShiftGabEchos(
      inputAsVariable,
      config_,
      firstDelay,
      rt60,
      numWalls,
      &randomEngine_,
      &randomUnit_,
      debug);

  augmented.host(output->data());
}

constexpr int kMaxDebugElementsInFileName = 6;
void Reverberation::randomShift(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debug,
    std::stringstream* debugFileName) {
  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable reverb(inputAsAfArray.copy(), false);

  const int numWalls = randomNumWalls_(randomEngine_);
  const float maxDistanceToWallInMeters_ =
      randomDistanceToWallInMeters_(randomEngine_);

  std::vector<float> debugDist;
  std::vector<float> debugAbsorb;

  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
    fl::Variable echo(inputAsAfArray.copy(), false);
    echo = echo / numWalls;

    const float distanceToWallInMeters =
        maxDistanceToWallInMeters_ * randomZeroToOne_(randomEngine_);
    const float absorptionCoefficient =
        randomAbsorptionCoefficient_(randomEngine_);

    const float delay = distanceToWallInMeters / kSpeedOfSoundMeterPerSec;
    const float rt60 = calcRt60(distanceToWallInMeters, absorptionCoefficient);
    const float attenuation = pow(10, -3 * delay / rt60);
    const size_t padFrames = delay * config_.sampleRate_;
    size_t zeroPadding = 0;

    if (debug) {
      *debug << "distanceToWallInMeters=" << distanceToWallInMeters
             << " absorptionCoefficient=" << absorptionCoefficient << std::endl;
      debugDist.push_back(distanceToWallInMeters);
      debugAbsorb.push_back(absorptionCoefficient);
    }

    while (true) {
      frac *= attenuation;
      echo = echo * attenuation;
      zeroPadding = zeroPadding + padFrames;
      if (frac < 1e-3 || zeroPadding >= echo.elements()) {
        break;
      }

      // Delay the echo in time by padding with zero on the left
      echo = fl::padding(
          echo,
          std::vector<std::pair<int, int>>({{zeroPadding, 0}}),
          /*val=*/0);

      // trim echo to length of reverb
      af::array& echoArray = echo.array();
      echoArray = echoArray(af::seq(1, reverb.elements()));

      reverb = reverb + echo;
    }
  }

  if (debugFileName) {
    std::vector<size_t> idx(debugDist.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort by distance from far to near
    std::stable_sort(
        idx.begin(), idx.end(), [&debugDist](size_t lhs, size_t rhs) {
          return debugDist[lhs] >= debugDist[rhs];
        });

    *debugFileName << "-walls-" << numWalls;
    idx.resize(kMaxDebugElementsInFileName);
    *debugFileName << "-dist";
    for (int i : idx) {
      *debugFileName << '-' << std::setprecision(2) << debugDist[i];
    }
    *debugFileName << "-absrb";
    for (int j : idx) {
      *debugFileName << '-' << std::setprecision(2) << debugAbsorb[j];
    }
  }

  reverb.host(output->data());
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

void Reverberation::conv1d(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debug,
    std::stringstream* debugFileName) {
  fl::Variable kernel = generateImpulseResponseKernel(
      config_.lengthMilliseconds_ * (config_.sampleRate_ / 1000),
      &randomEngine_,
      &randomUnit_,
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
  std::stringstream debugFileName;

  float absorptionCoefficientMin_ = 3.3;
  float absorptionCoefficientMax_ = 0.77;
  float distanceToWallInMetersMin_ = 2.0;
  float distanceToWallInMetersMax_ = 10.0;
  size_t numWallsMin_ = 3;
  float jitter_ = 0.1;

  if (config_.debugLevel_ > 1) {
    debug << "Reverberation::augmentImpl(signal->size()=" << signal->size()
          << ") config={" << config_.prettyString() << "}" << std::endl;
    if (config_.debugLevel_ > 2) {
      debugFileName << config_.debugOutputPath_ << '/'
                    << config_.debugOutputFilePrefix_;
    }
  }

  std::vector<float> augmented(signal->size(), 0);
  // conv1d(*signal, &augmented, &debug);
  // randomShift(*signal, &augmented, &debug, &debugFileName);
  randomShiftGab(*signal, &augmented, &debug, &debugFileName);

  if (config_.debugLevel_ > 1) {
    std::cout << debug.str() << std::endl;
    if (config_.debugLevel_ > 2) {
      static size_t idx = 0;
      ++idx;
      debugFileName << "-idx-" << std::setfill('0') << std::setw(4) << idx;
      const std::string metaDataFilename = debugFileName.str() + ".txt";
      const std::string audioFilename = debugFileName.str() + ".flac";
      const std::string dryAudioFilename = debugFileName.str() + "-dry.flac";

      debug << "saving augmented file=" << audioFilename << std::endl;
      if (config_.debugLevel_ > 3) {
        debug << "saving dry file=" << dryAudioFilename << std::endl;

        std::ofstream infoFile(metaDataFilename);
        if (infoFile.is_open()) {
          infoFile << debug.str() << std::endl;
        }

        saveSound(
            dryAudioFilename,
            *signal,
            16000,
            1,
            w2l::SoundFormat::FLAC,
            w2l::SoundSubFormat::PCM_16);
      }

      saveSound(
          audioFilename,
          augmented,
          16000,
          1,
          w2l::SoundFormat::FLAC,
          w2l::SoundSubFormat::PCM_16);
    }
    std::cout << debug.str() << std::endl;
  }
  signal->swap(augmented);
}

} // namespace augmentation
} // namespace w2l
