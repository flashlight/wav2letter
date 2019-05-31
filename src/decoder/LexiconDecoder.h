/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "decoder/Decoder.h"
#include "decoder/LM.h"
#include "decoder/Trie.h"

namespace w2l {
/**
 * LexiconDecoderState stores information for each hypothesis in the beam.
 */
struct LexiconDecoderState {
  LMStatePtr lmState_; // Language model state
  const TrieNode* lex_; // Trie node in the lexicon
  const LexiconDecoderState* parent_; // Parent hypothesis
  float score_; // Score so far
  int token_; // Label of token
  const TrieLabel* word_; // Label of word (-1 if incomplete)
  bool prevBlank_;

  LexiconDecoderState(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const float score,
      const int token,
      const TrieLabel* word,
      const bool prevBlank = false)
      : lmState_(lmState),
        lex_(lex),
        parent_(parent),
        score_(score),
        token_(token),
        word_(word),
        prevBlank_(prevBlank) {}

  LexiconDecoderState()
      : lmState_(nullptr),
        lex_(nullptr),
        parent_(nullptr),
        score_(0),
        token_(-1),
        word_(nullptr),
        prevBlank_(false) {}

  int getWord() const {
    return word_ ? word_->usr_ : -1;
  }

  bool isComplete() const {
    return !parent_ || parent_->word_;
  }
};

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| + unkScore_ *
 * |W_unknown| - silWeight_ * |{i| pi_i = <sil>}|
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
      const TriePtr lexicon,
      const LMPtr lm,
      const int sil,
      const int blank,
      const TrieLabelPtr unk,
      const std::vector<float>& transitions)
      : Decoder(opt),
        lexicon_(lexicon),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank),
        unk_(unk),
        nCandidates_(0) {
    candidates_.reserve(kBufferBucketSize);
  }

  void decodeBegin() override;

  void decodeEnd() override;

  int nHypothesis() const;

  int nDecodedFramesInBuffer() const;

  void prune(int lookBack = 0) override;

  DecodeResult getBestHypothesis(int lookBack = 0) const override;

  std::vector<DecodeResult> getAllFinalHypothesis() const override;

 protected:
  TriePtr lexicon_;
  LMPtr lm_;
  std::vector<float> transitions_;

  std::vector<LexiconDecoderState>
      candidates_; // All the hypothesis candidates (can be larger than
                   // beamsize) for the current frame
  std::vector<LexiconDecoderState*>
      candidatePtrs_; // This vector is only used when sort the candidates_, so
                      // instead of moving around objects, we only need to sort
                      // pointers
  float candidatesBestScore_;
  int sil_; // Index of silence label
  int blank_; // Index of blank label (for CTC)
  TrieLabelPtr unk_; // Trie label for unknown word
  std::unordered_map<int, std::vector<LexiconDecoderState>>
      hyp_; // Vector of hypothesis for all the frames so far
  int nCandidates_; // Total number of candidates in candidates_. Note that
                    // candidates is not always equal to candidates_.size()
                    // since we do not refresh the buffer for candidates_ in
                    // memory through out the whole decoding process.
  int nDecodedFrames_; // Total number of decoded frames.
  int nPrunedFrames_; // Total number of pruned frames from hyp_.

  void candidatesReset();

  void candidatesAdd(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const LexiconDecoderState* parent,
      const float score,
      const int token,
      const TrieLabel* label,
      const bool prevBlank);

  void candidatesStore(
      std::vector<LexiconDecoderState>& nextHyp,
      const bool isSort);

  virtual int mergeCandidates(const int size) = 0;
};

} // namespace w2l
