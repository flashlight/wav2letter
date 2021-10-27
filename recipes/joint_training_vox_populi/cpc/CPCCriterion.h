/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

//#include "flashlight/common/FlashlightUtils.h"
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/pkg/speech/criterion/Defines.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"

namespace w2l {

constexpr const char* kCPCCriterion = "cpc";

namespace detail {}

void PartialLoading(
    int n_layers,
    std::shared_ptr<fl::Sequential> net0,
    std::shared_ptr<fl::Sequential> net);

class CPCCriterion : public fl::pkg::speech::SequenceCriterion {
 public:
  CPCCriterion(
      int nEncoder,
      int nContext,
      int nMutual,
      int nOffset,
      int nUnits,
      int nPieces,
      int nNegative,
      int nBuffer,
      float temperature);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  af::array viterbiPath(
      const af::array& input,
      const af::array& inputSize = af::array()) override;

  std::string prettyString() const override;

  fl::Variable
  getMask(const fl::Variable& input, float mask_prob, int mask_length);
  float numMask() {
    return sum(masked_, {1, 2}).scalar<float>();
  }
  fl::Variable getMaskEmbedding() {
    return params_[0];
  }

 private:
  af::array getRandomIntegers(int N);
  fl::Variable getNegativeSamples(const fl::Variable& inp);

  std::shared_ptr<fl::Linear> mutualLinear(int k) const {
    return std::static_pointer_cast<fl::Linear>(module(k));
  }

  int nEncoder_;
  int nContext_;
  int nMutual_;
  int nOffset_;
  int nUnits_;
  int nPieces_;
  int nNegative_;
  int nBuffer_;
  float temperature_;
  fl::Variable masked_;
  af::array not_masked_;

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      nEncoder_,
      nContext_,
      nMutual_,
      nOffset_,
      nUnits_,
      nPieces_,
      nNegative_,
      nBuffer_,
      temperature_)

  CPCCriterion() = default;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::CPCCriterion)
CEREAL_CLASS_VERSION(w2l::CPCCriterion, 3)
