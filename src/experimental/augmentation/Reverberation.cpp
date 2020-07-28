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
     << " lengthMilliseconds_=" << lengthMilliseconds_
     << " backend_=" << static_cast<int>(backend_);
  return ss.str();
}

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
  }
  if (debugFilename) {
    const float dist = (firstDelay * kSpeedOfSoundMeterPerSec);
    *debugFilename << "-dist-" << std::setprecision(2) << dist << "-absrb-"
                   << std::setprecision(3) << absorptionCoefficient(dist, rt60)
                   << "-echos-" << numWalls << "-jitter-"
                   << reverbConfig_.jitter_;
  }

  if (reverbConfig_.backend_ == Reverberation::Config::Backend::GPU_GAB) {
    std::cout << "randomShiftGabGpu()" << std::endl;
    randomShiftGabGpu(input, output, firstDelay, rt60, numWalls);
  } else {
    std::cout << "randomShiftGabCpu()" << std::endl;
    randomShiftGabCpu(input, output, firstDelay, rt60, numWalls);
  }
}

void Reverberation::randomShiftGabGpu(
    const std::vector<float>& input,
    std::vector<float>* output,
    float firstDelay,
    float rt60,
    int numWalls) {
  const af::array inputAsAfArray(input.size(), input.data());
  fl::Variable inputAsVariable(inputAsAfArray, false);

  fl::Variable reverb(inputAsVariable.array().copy(), false);
  int memUsedForOne = 0;
  for (int i = 0; i < numWalls; ++i) {
    float frac = 1.0;
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
      std::cout << "jitter=" << jitter << " delay=" << delay
                << " totalDelay=" << totalDelay
                << " attenuationRandomness=" << attenuationRandomness
                << " attenuation=" << attenuation << " frac=" << frac
                << std::endl;

      if ((frac < 1e-3) || (totalDelay >= input.size())) {
        break;
      }

      // To shift the echo to the current totalDelay frame:
      // echo = input.subset(all elements except the last #totalDelay elements)
      // echo *= (attenuation / numWalls)
      // echo.pad(with #totalDelay zeros)
      // reverb += echo

      const int numElementsAfterPadding = input.size() - totalDelay;
      std::cout << "input.size()=" << input.size()
                << " reverb.elements()=" << reverb.elements()
                << " frac=" << frac << " totalDelay=" << totalDelay
                << " numElementsAfterPadding=" << numElementsAfterPadding
                << std::endl;
      fl::Variable echo = inputAsVariable(af::seq(1, numElementsAfterPadding)) *
          (attenuation / numWalls);
      std::cout << "line=" << __LINE__ << " echo.elements()=" << echo.elements()
                << std::endl;
      echo = fl::padding(
          echo,
          std::vector<std::pair<int, int>>({{totalDelay, 0}}),
          /*val=*/0);
      std::cout << "line=" << __LINE__ << " echo.elements()=" << echo.elements()
                << std::endl;
      reverb = reverb + echo;
      std::cout << "line=" << __LINE__
                << " reverb.elements()=" << reverb.elements() << std::endl;
      // reverb.eval();
    }
  }

  reverb.eval();
  reverb.host(output);
}

void Reverberation::randomShiftGabCpu(
    const std::vector<float>& input,
    std::vector<float>* output,
    float firstDelay,
    float rt60,
    int numWalls) {
  const int inputSize = input.size();
  *output = input;
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
      // output += shift(echo, totalDelay)
      for (int i = 0; i < overlapSize; ++i) {
        output->operator[](i + totalDelay) += echo[i] * multiplier;
      }
    }
  }
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
    const float decrasingAmplitude = amplitudeDiffPerFrame * i;
    const float randomImpulse = (*random_impulse)(*randomEngine);
    impulseResponseFrames[i] = decrasingAmplitude * randomImpulse;
  }
}

} // namespace

void Reverberation::conv1d(
    const std::vector<float>& input,
    std::vector<float>* output,
    std::stringstream* debugMsg,
    std::stringstream* debugFilename) {
  const size_t halfKernelSize =
      (reverbConfig_.lengthMilliseconds_ / 1000.0) * reverbConfig_.sampleRate_;
  const size_t kernelsize = (halfKernelSize * 2) + 1;
  const size_t center = halfKernelSize + 1;
  std::vector<float> kernelVector(kernelsize, 0);
  kernelVector[center] = 1.0;
  for (size_t i = 0; i < center; i += 100) {
    kernelVector[i] = 0.2;
  }

  af::array kernelArray(kernelVector.size(), kernelVector.data());
  fl::Variable kernel(kernelArray, false);
  fl::Conv2D reverbConv(kernel);
  if (debugMsg) {
    *debugMsg << "reverbConv=" << reverbConv.prettyString() << std::endl;
  }
  if (debugFilename) {
    *debugFilename << "-conv";
  }

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

  if (reverbConfig_.backend_ == Reverberation::Config::Backend::GPU_GAB ||
      reverbConfig_.backend_ == Reverberation::Config::Backend::GPU_GAB) {
    std::cout << "randomShiftGab()" << std::endl;
    randomShiftGab(*signal, &augmented, debugMsg, debugFilename);
  } else {
    std::cout << "conv1d()" << std::endl;
    conv1d(*signal, &augmented, debugMsg, debugFilename);
  }
  // randomShift(*signal, &augmented, &debugMsg, &debugFilename);

  signal->swap(augmented);
}

} // namespace augmentation
} // namespace w2l
