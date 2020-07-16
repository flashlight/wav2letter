/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include "data/Sound.h"
#include "experimental/augmentation/AudioLoader.h"

namespace w2l {

/**
 * Performs speed pertubation to the audio input by a fixed rate.
 *
 * @param input a float vector of input of shape channels X T (row major)
 * @param speed speed factor to be used
 * @param channels number of channels in the input audio
 * @return a float vector of shape channels X T' where T' ~= T / speed
 */
std::vector<float>
speedAug(const std::vector<float>& input, double speed, int channels = 1);

namespace augmentation {

class SoundEffect {
 public:
  virtual ~SoundEffect() = default;

  void operator()(
      std::vector<float>* signal,
      std::stringstream* debugSaveAugmentedFileName = nullptr) {
    return apply(signal, debugSaveAugmentedFileName);
  }

  virtual void apply(
      std::vector<float>* signal,
      std::stringstream* debugSaveAugmentedFileName = nullptr) = 0;

  virtual void enable(bool isEnable) {
    if (isEnable_ != isEnable) {
      std::cout << name() << "::enable(isEnable=" << isEnable << ")"
                << std::endl;
    }
    isEnable_ = isEnable;
  }

  virtual std::string prettyString() const = 0;
  virtual std::string name() const = 0;

 protected:
  bool isEnable_ = false;
};

class SoundEffectChain : public SoundEffect {
 public:
  ~SoundEffectChain() override {}

  void apply(
      std::vector<float>* signal,
      std::stringstream* debugSaveAugmentedFileName = nullptr) override {
    for (std::shared_ptr<SoundEffect>& effect : soundEffects_) {
      effect->apply(signal, debugSaveAugmentedFileName);
    }
  }

  void add(std::shared_ptr<SoundEffect> SoundEffect) {
    soundEffects_.push_back(SoundEffect);
  }

  void enable(bool isEnable) override {
    for (std::shared_ptr<SoundEffect>& effect : soundEffects_) {
      effect->enable(isEnable);
    }
    isEnable_ = isEnable;
  }

  std::string prettyString() const override {
    std::stringstream ss;
    ss << '{' << std::endl;
    for (const std::shared_ptr<SoundEffect>& effect : soundEffects_) {
      ss << effect->prettyString() << std::endl;
    }
    ss << '}';
    return ss.str();
  }

  std::string name() const override {
    return "SoundEffectChain";
  }

 private:
  std::vector<std::shared_ptr<SoundEffect>> soundEffects_;
};

class AudioAugmenter {
 public:
  virtual ~AudioAugmenter() = default;

  void augment(
      std::vector<float>* signal,
      std::stringstream* debugSaveAugmentedFileName) {
    if (isEnable_) {
      augmentImpl(signal, debugSaveAugmentedFileName);
    }
  }

  virtual void enable(bool isEnable) {
    if (isEnable_ != isEnable) {
      std::cout << "AudioAugmenter::enable(isEnable=" << isEnable << ")"
                << std::endl;
    }
    isEnable_ = isEnable;
  }

  std::string virtual prettyString() const = 0;

 protected:
  virtual void augmentImpl(
      std::vector<float>* signal,
      std::stringstream* debugSaveAugmentedFileName) = 0;

  bool isEnable_ = false;
};

} // namespace augmentation
} // namespace w2l
