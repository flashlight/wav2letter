/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "libraries/decoder/Decoder.h"
#include "libraries/lm/LM.h"

namespace w2l {
/**
 * LexiconFreeDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconFreeDecoderState {
  LMStatePtr lmState; // Language model state
  const LexiconFreeDecoderState* parent; // Parent hypothesis
  float score; // Score so far
  int token; // Label of token
  bool prevBlank; // If previous hypothesis is blank (for CTC only)

  LexiconFreeDecoderState(
      const LMStatePtr& lmState,
      const LexiconFreeDecoderState* parent,
      const float score,
      const int token,
      const bool prevBlank = false)
      : lmState(lmState),
        parent(parent),
        score(score),
        token(token),
        prevBlank(prevBlank) {}

  LexiconFreeDecoderState()
      : lmState(nullptr),
        parent(nullptr),
        score(0),
        token(-1),
        prevBlank(false) {}

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
      const LMPtr& lm,
      const int sil,
      const int blank,
      const std::vector<float>& transitions)
      : Decoder(opt),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank) {}

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

  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::vector<LexiconFreeDecoderState> candidates_;

  // This vector is designed for efficient sorting and merging the candidates_,
  // so instead of moving around objects, we only need to sort pointers
  std::vector<LexiconFreeDecoderState*> candidatePtrs_;

  // Best candidate score of current frame
  float candidatesBestScore_;

  // Index of silence label
  int sil_;

  // Index of blank label (for CTC)
  int blank_;

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconFreeDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.

  // Reset candidates buffer for decoding a new input frame
  void candidatesReset();

  // Add a new candidate to the buffer
  void candidatesAdd(
      const LMStatePtr& lmState,
      const LexiconFreeDecoderState* parent,
      const float score,
      const int token,
      const bool prevBlank);

  // Merge and sort candidates proposed in the current frame and place them into
  // the `hyp_` buffer
  void candidatesStore(
      std::vector<LexiconFreeDecoderState>& nextHyp,
      const bool isSort);

  // Merge hypothesis getting into same state from different path
  void mergeCandidates();
};

} // namespace w2l
