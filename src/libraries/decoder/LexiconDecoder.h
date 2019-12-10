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
#include "libraries/decoder/Trie.h"
#include "libraries/lm/LM.h"

namespace w2l {
/**
 * LexiconDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconDecoderState {
  LMStatePtr lmState; // Language model state
  const TrieNode* lex; // Trie node in the lexicon
  const LexiconDecoderState* parent; // Parent hypothesis
  double score; // Score so far
  int token; // Label of token
  int word; // Label of word (-1 if incomplete)
  bool prevBlank; // If previous hypothesis is blank (for CTC only)

  LexiconDecoderState(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const double score,
      const int token,
      const int word,
      const bool prevBlank = false)
      : lmState(lmState),
        lex(lex),
        parent(parent),
        score(score),
        token(token),
        word(word),
        prevBlank(prevBlank) {}

  LexiconDecoderState()
      : lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        score(0),
        token(-1),
        word(-1),
        prevBlank(false) {}

  int getWord() const {
    return word;
  }

  bool isComplete() const {
    return !parent || parent->word >= 0;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| + unkScore_ *
 * |W_unknown| + silScore_ * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) acoustic model
 * score of the transcription W. Note that the lexicon is used to limit the
 * search space and all candidate words are generated from it if unkScore is
 * -inf, otherwise <UNK> will be generated for OOVs.
 */
class LexiconDecoder : public Decoder {
 public:
  LexiconDecoder(
      const DecoderOptions& opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int sil,
      const int blank,
      const int unk,
      const std::vector<float>& transitions)
      : Decoder(opt),
        lexicon_(lexicon),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank),
        unk_(unk) {}

  void decodeBegin() override;

  void decodeEnd() override;

  int nHypothesis() const;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  TriePtr lexicon_;
  LMPtr lm_;
  std::vector<float> transitions_;

  // All the hypothesis new candidates (can be larger than beamsize) proposed
  // based on the ones from previous frame
  std::vector<LexiconDecoderState> candidates_;

  // This vector is designed for efficient sorting and merging the candidates_,
  // so instead of moving around objects, we only need to sort pointers
  std::vector<LexiconDecoderState*> candidatePtrs_;

  // Best candidate score of current frame
  double candidatesBestScore_;

  // Index of silence label
  int sil_;

  // Index of blank label (for CTC)
  int blank_;

  // Index of unknown word
  int unk_;

  // Vector of hypothesis for all the frames so far
  std::unordered_map<int, std::vector<LexiconDecoderState>> hyp_;

  // These 2 variables are used for online decoding, for hypothesis pruning
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.

  // Reset candidates buffer for decoding a new input frame
  void candidatesReset();

  // Add a new candidate to the buffer
  void candidatesAdd(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const double score,
      const int token,
      const int label,
      const bool prevBlank);

  // Merge and sort candidates proposed in the current frame and place them into
  // the `hyp_` buffer
  void candidatesStore(
      std::vector<LexiconDecoderState>& nextHyp,
      const bool isSort);

  // Merge hypothesis getting into same state from different path
  virtual void mergeCandidates() = 0;
};

} // namespace w2l
