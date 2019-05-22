/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <random>

#include <flashlight/flashlight.h>

namespace w2l {

/**
 * Implementation of SpecAugment: A Simple Data Augmentation Method
 * for Automatic Speech Recognition - https://arxiv.org/pdf/1904.08779.pdf
 *
 * We assume time axis is the 0th dimension, and freq axis is the 1st dimension
 * for the  input array
 *
 * Example policies        tWarpW    fMaskF    nFMask tMaskT    tMaskP   nTMask
 * LibriSpeech basic (LB)    80        27        1     100       1.0       1
 * LibriSpeech double (LD)   80        27        2     100       1.0       2
 * Switchboard mild (SM)     40        15        2      70       0.2       2
 * Switchboard strong (SS)   40        27        2      70       0.2       2
 **/
class SpecAugment : public fl::UnaryModule {
 public:
  enum class MaskingStrategy {
    ZERO = 0,
    GLOBAL_MEAN = 1,
    // TODO - add support for mean along time, freq axes
  };

  SpecAugment(
      int tWarpW,
      int fMaskF,
      int nFMask,
      int tMaskT,
      float tMaskP,
      int nTMask,
      MaskingStrategy mStrategy = MaskingStrategy::ZERO);

  fl::Variable forward(const fl::Variable& input) override;

  FL_SAVE_LOAD_WITH_BASE(
      fl::UnaryModule,
      timeWarpW_,
      freqMaskF_,
      numFreqMask_,
      timeMaskT_,
      timeMaskP_,
      numTimeMask_,
      maskStrategy_)

  std::string prettyString() const override;

 private:
  // Time Warping - NOT SUPPORTED CURRENTLY
  //  Use timeWarpW_ = 0 to disable this
  int timeWarpW_;

  // Frequency Masking
  //  Use freqMaskF_ = 0 to disable this
  int freqMaskF_;
  int numFreqMask_;

  // Time Masking
  //  Use timeMaskT_ = 0 to disable this
  int timeMaskT_;
  float timeMaskP_;
  int numTimeMask_;

  std::mt19937 eng_{0};
  MaskingStrategy maskStrategy_;

  int generateRandomInt(int low, int high);

  SpecAugment() = default;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::SpecAugment)
