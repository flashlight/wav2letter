/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <unordered_map>

#include "decoder/Decoder.h"
#include "decoder/LM.h"

namespace w2l {
typedef std::shared_ptr<void> AMStatePtr;
typedef std::function<
    std::pair<std::vector<std::vector<float>>, std::vector<AMStatePtr>>(
        const float*,
        const int,
        const int,
        const std::vector<int>&,
        const std::vector<AMStatePtr>&,
        int&)>
    AMUpdateFunc;

/**
 * Seq2SeqDecoderState stores information for each hypothesis in the beam.
 */
struct Seq2SeqDecoderState {
  LMStatePtr lmState_; // Language model state
  const Seq2SeqDecoderState* parent_; // Parent hypothesis
  float score_; // Score so far
  int token_; // Label of token
  AMStatePtr amState_; // Acoustic model state

  Seq2SeqDecoderState(
      const LMStatePtr& lmState,
      const Seq2SeqDecoderState* parent,
      const float score,
      const int token,
      const AMStatePtr& amState = nullptr)
      : lmState_(lmState),
        parent_(parent),
        score_(score),
        token_(token),
        amState_(amState) {}

  Seq2SeqDecoderState()
      : lmState_(nullptr),
        parent_(nullptr),
        score_(0),
        token_(-1),
        amState_(nullptr) {}

  int getWord() const {
    return -1;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known|
 *
 * where P_{lm}(W) is the language model score. Note that the transcription is
 * made up of word-pieces, no real `word` is included.
 *
 * TODO: Doesn't support online decoding now.
 *
 */
class Seq2SeqDecoder : public Decoder {
 public:
  Seq2SeqDecoder(
      const DecoderOptions& opt,
      const LMPtr lm,
      const int eos,
      AMUpdateFunc amUpdateFunc,
      const int maxOutputLength,
      const float hardSelection,
      const float softSelection)
      : Decoder(opt),
        lm_(lm),
        eos_(eos),
        amUpdateFunc_(amUpdateFunc),
        maxOutputLength_(maxOutputLength),
        hardSelection_(hardSelection),
        softSelection_(softSelection) {
    candidates_.reserve(kBufferBucketSize);
  }

  void decodeStep(const float* emissions, int T, int N) override;

  void prune(int lookBack = 0) override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  LMPtr lm_;
  int eos_;
  AMUpdateFunc amUpdateFunc_;
  std::vector<Seq2SeqDecoderState> completedCandidates;
  std::vector<int> rawY;
  std::vector<AMStatePtr> rawPrevStates;
  int maxOutputLength_;
  float hardSelection_;
  float softSelection_;

  std::vector<Seq2SeqDecoderState> candidates_;
  std::vector<Seq2SeqDecoderState*> candidatePtrs_;
  float candidatesBestScore_;
  int nCandidates_;

  std::unordered_map<int, std::vector<Seq2SeqDecoderState>> hyp_;

  std::vector<Seq2SeqDecoderState> completedCandidates_;

  void candidatesReset();

  void candidatesAdd(
      const LMStatePtr& lmState,
      const Seq2SeqDecoderState* parent,
      const float score,
      const int token,
      const AMStatePtr& amState);

  void candidatesStore(
      std::vector<Seq2SeqDecoderState>& nextHyp,
      const bool isSort);

  int mergeCandidates(const int size);
};

} // namespace w2l
