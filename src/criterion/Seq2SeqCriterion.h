/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include "SequenceCriterion.h"
#include "common/Utils.h"
#include "criterion/attention/attention.h"
#include "criterion/attention/window.h"

namespace w2l {

class Seq2SeqCriterion : public SequenceCriterion {
 public:
  struct DecoderState {
    fl::Variable alpha;
    fl::Variable hidden;
    fl::Variable summary;
    int step;
    DecoderState() : step(0) {}
  };

  struct CandidateHypo {
    float score;
    std::vector<int> path;
    DecoderState state;
    explicit CandidateHypo() : score(0.0) {
      path.resize(0);
    };
    CandidateHypo(float score_, std::vector<int> path_, DecoderState state_)
        : score(score_), path(path_), state(state_) {}
  };

  Seq2SeqCriterion(
      int nClass,
      int hiddenDim,
      int eos,
      int maxDecoderOutputLen,
      std::shared_ptr<AttentionBase> attention,
      std::shared_ptr<WindowBase> window = nullptr,
      bool trainWithWindow = false,
      int pctTeacherForcing = 100,
      bool useSequentialDecoder = false,
      double labelSmooth = 0.0,
      bool inputFeeding = false);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  /* Next step predictions are based on the target at
   * the previous time-step so this function should only
   * be used for training purposes. */
  std::pair<fl::Variable, fl::Variable> decoder(
      const fl::Variable& input,
      const fl::Variable& target);

  std::pair<fl::Variable, fl::Variable> vectorizedDecoder(
      const fl::Variable& input,
      const fl::Variable& target);

  af::array viterbiPath(const af::array& input) override;

  std::pair<af::array, fl::Variable> viterbiPathBase(
      const af::array& input,
      bool saveAttn);

  std::vector<CandidateHypo> beamSearch(
      const af::array& input,
      std::vector<Seq2SeqCriterion::CandidateHypo> beam,
      int beamSize,
      int maxLen);

  std::vector<int> beamPath(const af::array& input, int beamSize = 10);

  std::string prettyString() const override;

  std::shared_ptr<fl::Embedding> embedding() {
    return std::static_pointer_cast<fl::Embedding>(module(0));
  }

  std::shared_ptr<fl::RNN> decodeRNN() {
    return std::static_pointer_cast<fl::RNN>(module(1));
  }

  std::shared_ptr<fl::Linear> linearOut() {
    return std::static_pointer_cast<fl::Linear>(module(2));
  }

  std::shared_ptr<AttentionBase> attention() {
    return std::static_pointer_cast<AttentionBase>(module(3));
  }

  fl::Variable startEmbedding() {
    return params_.back();
  }

 private:
  int eos_;
  int maxDecoderOutputLen_;
  std::shared_ptr<WindowBase> window_;
  bool trainWithWindow_;
  int pctTeacherForcing_;
  bool useSequentialDecoder_;
  double labelSmooth_;
  bool inputFeeding_;
  int nClass_;

  FL_SAVE_LOAD_WITH_BASE(
      SequenceCriterion,
      eos_,
      maxDecoderOutputLen_,
      window_,
      trainWithWindow_,
      pctTeacherForcing_,
      useSequentialDecoder_,
      labelSmooth_,
      inputFeeding_,
      nClass_)

  Seq2SeqCriterion() = default;

  std::pair<fl::Variable, DecoderState> decodeStep(
      const fl::Variable& xEncoded,
      const fl::Variable& y,
      const DecoderState& instate);
};

w2l::Seq2SeqCriterion buildSeq2Seq(int numClasses, int eosIdx);

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::Seq2SeqCriterion)
