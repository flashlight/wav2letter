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

#include "libraries/decoder/Decoder.h"
#include "libraries/decoder/Trie.h"
#include "libraries/lm/LM.h"

namespace w2l {
using AMStatePtr = std::shared_ptr<void>;
using AMUpdateFunc = std::function<
    std::pair<std::vector<std::vector<float>>, std::vector<AMStatePtr>>(
        const float*,
        const int,
        const int,
        const std::vector<int>&,
        const std::vector<AMStatePtr>&,
        int&)>;

/**
 * LexiconSeq2SeqDecoderState stores information for each hypothesis in the
 * beam.
 */
struct LexiconSeq2SeqDecoderState {
  double score; // Accumulated total score so far
  LMStatePtr lmState; // Language model state
  const TrieNode* lex;
  const LexiconSeq2SeqDecoderState* parent; // Parent hypothesis
  int token; // Label of token
  int word;
  AMStatePtr amState; // Acoustic model state

  double amScore; // Accumulated AM score so far
  double lmScore; // Accumulated LM score so far

  LexiconSeq2SeqDecoderState(
      const double score,
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconSeq2SeqDecoderState* parent,
      const int token,
      const int word,
      const AMStatePtr& amState,
      const double amScore = 0,
      const double lmScore = 0)
      : score(score),
        lmState(lmState),
        lex(lex),
        parent(parent),
        token(token),
        word(word),
        amState(amState),
        amScore(amScore),
        lmScore(lmScore) {}

  LexiconSeq2SeqDecoderState()
      : score(0),
        lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        token(-1),
        word(-1),
        amState(nullptr),
        amScore(0.),
        lmScore(0.) {}

  int compareNoScoreStates(const LexiconSeq2SeqDecoderState* node) const {
    int lmCmp = lmState->compare(node->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0 ? 1 : -1;
    } else if (lex != node->lex) {
      return lex > node->lex ? 1 : -1;
    } else if (token != node->token) {
      return token > node->token ? 1 : -1;
    }
    return 0;
  }

  int getWord() const {
    return word;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the token transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + eosScore_ * |W_last == EOS|
 *
 * where P_{lm}(W) is the language model score. The transcription W is
 * constrained by a lexicon. The language model may operate at word-level
 * (isLmToken=false) or token-level (isLmToken=true).
 *
 * TODO: Doesn't support online decoding now.
 *
 */
class LexiconSeq2SeqDecoder : public Decoder {
 public:
  LexiconSeq2SeqDecoder(
      const DecoderOptions& opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int eos,
      AMUpdateFunc amUpdateFunc,
      const int maxOutputLength,
      const bool isLmToken)
      : Decoder(opt),
        lm_(lm),
        lexicon_(lexicon),
        eos_(eos),
        amUpdateFunc_(amUpdateFunc),
        maxOutputLength_(maxOutputLength),
        isLmToken_(isLmToken) {}

  void decodeStep(const float* emissions, int T, int N) override;

  void prune(int lookBack = 0) override;

  int nDecodedFramesInBuffer() const override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  LMPtr lm_;
  TriePtr lexicon_;
  int eos_;
  AMUpdateFunc amUpdateFunc_;
  std::vector<int> rawY_;
  std::vector<AMStatePtr> rawPrevStates_;
  int maxOutputLength_;
  bool isLmToken_;

  std::vector<LexiconSeq2SeqDecoderState> candidates_;
  std::vector<LexiconSeq2SeqDecoderState*> candidatePtrs_;
  double candidatesBestScore_;

  std::unordered_map<int, std::vector<LexiconSeq2SeqDecoderState>> hyp_;
};

} // namespace w2l
