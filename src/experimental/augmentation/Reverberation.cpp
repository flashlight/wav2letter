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

Reverberation::Reverberation(
    const SoundEffect::Config& sfxConfig,
    const Reverberation::Config& reverbConfig)
    : SoundEffect(sfxConfig),
      reverbConfig_(reverbConfig),
      randomEngine_(sfxConfig_.randomSeed_),
      randomZeroToOne_(0.0, 1.0),
      randomUnit_(-1.0, 1.0),
      randomDecay_(
          calcRt60(
              reverbConfig_.distanceToWallInMetersMax_,
              reverbConfig_.absorptionCoefficientMin_),
          calcRt60(
              reverbConfig_.distanceToWallInMetersMin_,
              reverbConfig_.absorptionCoefficientMax_)),
      randomDelay_(
          reverbConfig_.distanceToWallInMetersMin_ / kSpeedOfSoundMeterPerSec,
          reverbConfig_.distanceToWallInMetersMax_ / kSpeedOfSoundMeterPerSec),
      randomAbsorptionCoefficient_(
          reverbConfig_.absorptionCoefficientMin_,
          reverbConfig_.absorptionCoefficientMax_),
      randomDistanceToWallInMeters_(
          reverbConfig_.distanceToWallInMetersMax_,
          reverbConfig_.distanceToWallInMetersMin_),
      randomNumWalls_(
          static_cast<int>(reverbConfig_.numWallsMin_),
          static_cast<int>(reverbConfig_.numWallsMax_)) {
  assert(reverbConfig_.numWallsMin_ <= reverbConfig_.numWallsMax_);
  std::cout << "Reverberation::Reverberation(config={"
            << reverbConfig_.prettyString() << "})" << std::endl;
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
     << " lengthMilliseconds_=" << lengthMilliseconds_;
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
    std::stringstream* debugMsg) {
  fl::Variable reverb(source.array().copy(), false);
  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
    fl::Variable echo(source.array().copy(), false);
    echo = echo / numWalls;
    size_t totalDelay = 0;

    while (frac > 1e-3 && totalDelay < source.elements()) {
      const float jitter = 1 + config.jitter_ * (*randomUnit)(*randomEngine);
      size_t delay = std::min(
          1UL + static_cast<size_t>(jitter * firstDelay * config.sampleRate_),
          static_cast<size_t>(source.elements()));
      totalDelay += delay;

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
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  const float firstDelay = randomDelay_(randomEngine_);
  const float rt60 = randomDecay_(randomEngine_);
  const int numWalls = randomNumWalls_(randomEngine_);
  if (debugMsg) {
    *debugMsg << "firstDelay=" << firstDelay << "("
              << (firstDelay * kSpeedOfSoundMeterPerSec) << "m)"
              << " rt60=" << rt60 << std::endl;
    if (debugFilename) {
      const float dist = (firstDelay * kSpeedOfSoundMeterPerSec);
      *debugFilename << "-dist-" << std::setprecision(2) << dist << "-absrb-"
                     << std::setprecision(3)
                     << absorptionCoefficient(dist, rt60) << "-echos-"
                     << numWalls << "-jitter-" << reverbConfig_.jitter_;
    }
  }

  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable inputAsVariable(inputAsAfArray, false);
  fl::Variable augmented = randomShiftGabEchos(
      inputAsVariable,
      reverbConfig_,
      firstDelay,
      rt60,
      numWalls,
      &randomEngine_,
      &randomUnit_,
      debugMsg);

  augmented.host(output->data());
}

void Reverberation::randomShiftGabGpu(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  const float firstDelay = randomDelay_(randomEngine_);
  const float rt60 = randomDecay_(randomEngine_);
  const int numWalls = randomNumWalls_(randomEngine_);
  if (debugMsg) {
    *debugMsg << "firstDelay=" << firstDelay << "("
              << (firstDelay * kSpeedOfSoundMeterPerSec) << "m)"
              << " rt60=" << rt60 << std::endl;
  }
  if (debugFilename) {
    const float dist = (firstDelay * kSpeedOfSoundMeterPerSec);
    *debugFilename << "-dist-" << std::setprecision(2) << dist << "-absrb-"
                   << std::setprecision(3) << absorptionCoefficient(dist, rt60)
                   << "-echos-" << numWalls << "-jitter-"
                   << reverbConfig_.jitter_;
  }

  const int inputSize = input.size();
  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable inputAsVariable(inputAsAfArray, false);

  fl::Variable reverb(inputAsVariable.array().copy(), false);
  int memUsedForOne = 0;
  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
    size_t totalDelay = 0;

    for (int echoCount = 1;; ++echoCount) {
      const float jitter =
          1 + reverbConfig_.jitter_ * (randomUnit_)(randomEngine_);
      size_t delay = 1UL +
          static_cast<size_t>(jitter * firstDelay * reverbConfig_.sampleRate_);
      totalDelay += delay;

      const float attenuationRandomness =
          1.0f + reverbConfig_.jitter_ * (randomUnit_)(randomEngine_);
      const float attenuation =
          pow(10, -3 * attenuationRandomness * firstDelay / rt60);
      frac *= attenuation;

      if ((frac < 1e-3) || (totalDelay >= inputSize)) {
        break;
      }

      const int memUsed =
          (totalDelay + inputAsVariable.elements()) * sizeof(float);
      memUsedForOne += memUsed;
      {
        std::stringstream ss;
        mtx_.lock();
        memUsed_ += memUsed;
        if (memUsed_ > maxMemUsed_) {
          maxMemUsed_ = memUsed_;
          ss << "maxMemUsed_=" << (maxMemUsed_ >> 10)
             << "KB memUsed_=" << (memUsed_ >> 10) << "KB"
             << " input.size()=" << input.size() << "("
             << (input.size() >> 10) * sizeof(float) << "KB)"
             << " totalDelay=" << totalDelay << "("
             << (totalDelay >> 10) * sizeof(float) << "KB)"
             << " applyCount_=" << applyCount_ << " echoCount=" << echoCount
             << " memUsedForOne=" << (memUsedForOne >> 10)
             << "KB=" << (memUsedForOne >> 20) << "MB"
             << " maxMemUsedForOne_=" << (maxMemUsedForOne_ >> 10)
             << "KB=" << (maxMemUsedForOne_ >> 20) << "MB" << std::endl;
        }
        mtx_.unlock();
        if (!ss.str().empty()) {
          std::cout << ss.str() << std::endl;
        }
        if (debugMsg) {
          // *debugMsg << ss.str();
        }
      }

      // echo = shit(input, totalDelay) * attenuation / numWalls;
      // reverb += echo;
      fl::Variable echo =
          fl::padding(
              inputAsVariable,
              std::vector<std::pair<int, int>>({{totalDelay, 0}}),
              /*val=*/0) *
          (attenuation / numWalls);
      // trim echo to length of reverb
      af::array& echoArray = echo.array();
      echoArray = echoArray(af::seq(1, reverb.elements()));

      reverb = reverb + echo;
      {
        mtx_.lock();
        memUsed_ -= memUsed;
        mtx_.unlock();
      }
    }
  }

  {
    mtx_.lock();
    if (memUsedForOne > maxMemUsedForOne_) {
      std::stringstream ss;
      ss << " memUsedForOne=" << (memUsedForOne >> 10)
         << "KB=" << (memUsedForOne >> 20) << "MB"
         << " maxMemUsedForOne_=" << (maxMemUsedForOne_ >> 10)
         << "KB=" << (maxMemUsedForOne_ >> 20) << "MB" << std::endl;
      std::cout << ss.str() << std::endl;

      maxMemUsedForOne_ = memUsedForOne;
      if (debugMsg) {
        // *debugMsg << ss.str();
      }
    }
    mtx_.unlock();
  }
  reverb.host(output->data());
}

void Reverberation::randomShiftGabCpu(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  const float firstDelay = randomDelay_(randomEngine_);
  const float rt60 = randomDecay_(randomEngine_);
  const int numWalls = randomNumWalls_(randomEngine_);
  if (debugMsg) {
    *debugMsg << "firstDelay=" << firstDelay << "("
              << (firstDelay * kSpeedOfSoundMeterPerSec) << "m)"
              << " rt60=" << rt60 << std::endl;
  }
  if (debugFilename) {
    const float dist = (firstDelay * kSpeedOfSoundMeterPerSec);
    *debugFilename << "-dist-" << std::setprecision(2) << dist << "-absrb-"
                   << std::setprecision(3) << absorptionCoefficient(dist, rt60)
                   << "-echos-" << numWalls << "-jitter-"
                   << reverbConfig_.jitter_;
  }

  const int inputSize = input.size();
  std::vector<float> reverb = input;
  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
    std::vector<float> echo = input;
    size_t totalDelay = 0;

    while (true) {
      const float jitter =
          1 + reverbConfig_.jitter_ * (randomUnit_)(randomEngine_);
      size_t delay = 1UL +
          static_cast<size_t>(jitter * firstDelay * reverbConfig_.sampleRate_);
      totalDelay += delay;

      const float attenuationRandomness =
          1.0f + reverbConfig_.jitter_ * (randomUnit_)(randomEngine_);
      const float attenuation =
          pow(10, -3 * attenuationRandomness * firstDelay / rt60);
      frac *= attenuation;

      if ((frac < 1e-3) || (totalDelay >= inputSize)) {
        break;
      }

      // echo *= attenuation

      const float multiplier = attenuation / numWalls;
      const int overlapSize = inputSize - totalDelay;

      // echo /= numWalls
      // echo *= attenuation
      // reverb += shift(echo, totalDelay)
      for (int i = 0; i < overlapSize; ++i) {
        reverb[i + totalDelay] += echo[i] * multiplier;
      }
    }
  }

  output->swap(reverb);
}

constexpr int kMaxDebugElementsInFileName = 6;
void Reverberation::randomShift(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
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
    const size_t padFrames = delay * reverbConfig_.sampleRate_;
    size_t zeroPadding = 0;

    if (debugMsg) {
      *debugMsg << "distanceToWallInMeters=" << distanceToWallInMeters
                << " absorptionCoefficient=" << absorptionCoefficient
                << std::endl;
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

  if (debugFilename) {
    std::vector<size_t> idx(debugDist.size());
    iota(idx.begin(), idx.end(), 0);

    // Sort by distance from far to near
    std::stable_sort(
        idx.begin(), idx.end(), [&debugDist](size_t lhs, size_t rhs) {
          return debugDist[lhs] >= debugDist[rhs];
        });

    *debugFilename << "-walls-" << numWalls;
    idx.resize(kMaxDebugElementsInFileName);
    *debugFilename << "-dist";
    for (int i : idx) {
      *debugFilename << '-' << std::setprecision(2) << debugDist[i];
    }
    *debugFilename << "-absrb";
    for (int j : idx) {
      *debugFilename << '-' << std::setprecision(2) << debugAbsorb[j];
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
    std::stringstream* debugMsg) {
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
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  fl::Variable kernel = generateImpulseResponseKernel(
      reverbConfig_.lengthMilliseconds_ * (reverbConfig_.sampleRate_ / 1000),
      &randomEngine_,
      &randomUnit_,
      sfxConfig_.debug_.debugLevel_,
      debugMsg);

  fl::Conv2D reverbConv(kernel, 1, 1, kernel.elements() / 2);
  *debugMsg << "reverbConv=" << reverbConv.prettyString() << std::endl;

  af::array signalArray(input.size(), input.data());
  fl::Variable signalVariable(signalArray, false);

  fl::Variable augmentedVariable = reverbConv.forward(signalVariable);
  augmentedVariable.eval();
  augmentedVariable.host(output->data());
}

void Reverberation::apply(
    std::vector<float>* signal,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  if (debugMsg) {
    *debugMsg << "Reverberation::apply(signal->size()=" << signal->size()
              << ")";
  }

  std::vector<float> augmented(signal->size(), 0);
  // conv1d(*signal, &augmented, &debugMsg);
  // randomShift(*signal, &augmented, &debugMsg, &debugFilename);
  // randomShiftGab(*signal, &augmented, &debugMsg, debugFilename);
  // randomShiftGabCpu(*signal, &augmented, debugMsg, debugFilename);
  randomShiftGabGpu(*signal, &augmented, debugMsg, debugFilename);

  signal->swap(augmented);
}

} // namespace augmentation
} // namespace w2l
