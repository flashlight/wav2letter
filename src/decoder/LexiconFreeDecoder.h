/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "Decoder.h"
#include "LM.h"

namespace w2l {
/**
 * LexiconFreeDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconFreeDecoderState {
  LMStatePtr lmState_; // Language model state
  const LexiconFreeDecoderState* parent_; // Parent hypothesis
  float score_; // Score so far
  int token_; // Label of token
  bool prevBlank_;

  LexiconFreeDecoderState(
      const LMStatePtr& lmState,
      const LexiconFreeDecoderState* parent,
      const float score,
      const int token,
      const bool prevBlank = false)
      : lmState_(lmState),
        parent_(parent),
        score_(score),
        token_(token),
        prevBlank_(prevBlank) {}

  LexiconFreeDecoderState()
      : lmState_(nullptr),
        parent_(nullptr),
        score_(0),
        token_(-1),
        prevBlank_(false) {}

  int getWord() const {
    return -1;
  }

  bool isComplete() const {
    return true;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| - silWeight_ *
 * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) acoustic model
 * score of the transcription W. We are allowed to generate words from all the
 * possible combination of tokens.
 */
class LexiconFreeDecoder : public Decoder {
 public:
  LexiconFreeDecoder(
      const DecoderOptions& opt,
      const LMPtr lm,
      const int sil,
      const int blank,
      const std::vector<float>& transitions,
      const std::unordered_map<int, int>& lmIndMap)
      : Decoder(opt),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank),
        nCandidates_(0),
        lmIndMap_(lmIndMap) {
    candidates_.reserve(kBufferBucketSize);
  }

  void decodeBegin() override;

  void decodeStep(const float* emissions, int T, int N) override;

  void decodeEnd() override;

  int nHypothesis() const;

  int nDecodedFramesInBuffer() const;

  void prune(int lookBack = 0) override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  LMPtr lm_;
  std::vector<float> transitions_;

  std::vector<LexiconFreeDecoderState>
      candidates_; // All the hypothesis candidates (can be larger than
                   // beamsize) for the current frame
  std::vector<LexiconFreeDecoderState*>
      candidatePtrs_; // This vector is only used when sort the candidates_, so
                      // instead of moving around objects, we only need to sort
                      // pointers
  float candidatesBestScore_;
  int sil_; // Index of silence label
  int blank_; // Index of blank label (for CTC)
  std::unordered_map<int, std::vector<LexiconFreeDecoderState>>
      hyp_; // Vector of hypothesis for all the frames so far
  int nCandidates_; // Total number of candidates in candidates_. Note that
                    // candidates is not always equal to candidates_.size()
                    // since we do not refresh the buffer for candidates_ in
                    // memory through out the whole decoding process.
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.

  std::unordered_map<int, int> lmIndMap_;

  void candidatesReset();

  void candidatesAdd(
      const LMStatePtr& lmState,
      const LexiconFreeDecoderState* parent,
      const float score,
      const int token,
      const bool prevBlank);

  void candidatesStore(
      std::vector<LexiconFreeDecoderState>& nextHyp,
      const bool isSort);

  int mergeCandidates(const int size);
};

} // namespace w2l
