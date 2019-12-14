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
  LMStatePtr lmState; // Language model state
  const TrieNode* lex;
  const LexiconSeq2SeqDecoderState* parent; // Parent hypothesis
  double score; // Score so far
  int token; // Label of token
  int word;
  AMStatePtr amState; // Acoustic model state

  LexiconSeq2SeqDecoderState(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconSeq2SeqDecoderState* parent,
      const double score,
      const int token,
      const int word,
      const AMStatePtr& amState)
      : lmState(lmState),
        lex(lex),
        parent(parent),
        score(score),
        token(token),
        word(word),
        amState(amState) {}

  LexiconSeq2SeqDecoderState()
      : lmState(nullptr),
        lex(nullptr),
        parent(nullptr),
        score(0),
        token(-1),
        word(-1),
        amState(nullptr) {}

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

  void candidatesReset();

  void candidatesAdd(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconSeq2SeqDecoderState* parent,
      const double score,
      const int token,
      const int word,
      const AMStatePtr& amState);

  void candidatesStore(
      std::vector<LexiconSeq2SeqDecoderState>& nextHyp,
      const bool isSort);

  void mergeCandidates();
};

} // namespace w2l
