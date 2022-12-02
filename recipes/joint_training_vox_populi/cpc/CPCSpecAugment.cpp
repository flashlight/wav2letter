/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>

#include <sstream>
#include <stdexcept>

#include "CPCSpecAugment.h"

namespace w2l {

CPCSpecAugment::CPCSpecAugment(
    int tWarpW,
    int fMaskF,
    int nFMask,
    int tMaskT,
    float tMaskP,
    int nTMask,
    MaskingStrategy mStrategy /* = MaskingStrategy::ZERO */)
    : timeWarpW_(tWarpW),
      freqMaskF_(fMaskF),
      numFreqMask_(nFMask),
      timeMaskT_(tMaskT),
      timeMaskP_(tMaskP),
      numTimeMask_(nTMask),
      maskStrategy_(mStrategy) {
  if (numFreqMask_ > 0 && freqMaskF_ <= 0) {
    throw std::invalid_argument("invalid arguments for frequency masking.");
  }
  if (numTimeMask_ > 0 && timeMaskT_ <= 0) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
  if (numTimeMask_ > 0 && (timeMaskP_ <= 0 || timeMaskP_ > 1.0)) {
    throw std::invalid_argument("invalid arguments for time masking.");
  }
}

int CPCSpecAugment::generateRandomInt(int low, int high) {
  std::uniform_int_distribution<int> uniformDist(low, high - 1);
  return uniformDist(eng_);
}

fl::Variable CPCSpecAugment::maskFunction(
    const fl::Variable& input,
    const fl::Variable& mask_emb,
    double mask_prob,
    int mask_length,
    int dim) {
  int T = input.dims(dim);
  int N = input.dims(2);

  int numMask = (mask_prob * T) / mask_length;
  auto mask = af::constant(0., af::dim4(T, N), f32);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < numMask; j++) {
      int startIdx = generateRandomInt(0, T);
      int endIdx = std::min(startIdx + mask_length - 1, T - 1);
      mask(af::seq(startIdx, endIdx), i) = 1.;
    }
  }

  // restrict by min len
  int minLen = af::min<int>(af::sum(mask, 0));
  auto maskMinLen = af::constant(0., af::dim4(T, N), f32);
  for (int i = 0; i < N; i++) {
    auto maskIdx = af::where(mask(af::span, i));
    auto tmp = af::randu(maskIdx.dims(0));
    af::array val, idx;
    af::sort(val, idx, tmp);
    idx = idx(af::seq(0, minLen - 1));
    maskIdx = maskIdx(idx);
    maskMinLen(maskIdx, i) = 1.;
  }
  mask = maskMinLen * 1;

  if (dim == 0) {
    mask = af::moddims(mask, af::dim4(T, 1, N));
  } else {
    mask = af::moddims(mask, af::dim4(1, T, N));
  }

  auto totalMask = tileAs(fl::Variable(mask, false), input.dims());
  auto maskEmbedding = tileAs(mask_emb, input.dims());
  auto inputMasked =
      input.as(f32) * (1 - totalMask) + maskEmbedding * totalMask;

  return inputMasked.as(input.type());
}

void CPCSpecAugment::setMaskEmbedding(const fl::Variable& input) {
  mask_emb_ = input * 1;
}

fl::Variable CPCSpecAugment::forward(const fl::Variable& input) {
  if (!train_) {
    return input;
  }

  auto output = maskFunction(input, mask_emb_, timeMaskP_, timeMaskT_, 1);
  // output = maskFunction(output, fl::constant(0.0, af::dim4(1)), 0.25, 64, 0);
  return output;
}

std::string CPCSpecAugment::prettyString() const {
  std::ostringstream ss;
  ss << "CPCSpecAugment ( ";
  ss << "W: " << timeWarpW_ << ", ";
  ss << "F: " << freqMaskF_ << ", ";
  ss << "mF: " << numFreqMask_ << ", ";
  // ss << "T: " << timeMaskT_ << ", ";
  ss << "p: " << timeMaskP_ << ", ";
  ss << "mT: " << numTimeMask_;
  ss << " )";
  return ss.str();
}
} // namespace w2l
