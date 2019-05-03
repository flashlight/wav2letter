/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "LM.hpp"
#include "Trie.hpp"

namespace w2l {

const int kBufferBucketSize = 65536;
const float kNegativeInfinity = -std::numeric_limits<float>::infinity();

enum class ModelType { ASG = 0, CTC = 1 };

/**
 * Decoder implements a beam seach decoder that finds the word transcription
 * W maximizing:
 *
 * AM(W) + lmWeight_ * log(P_{lm}(W)) + wordScore_ * |W_known| + unkScore_ *
 * |W_unknown| - silWeight_ * |{i| pi_i = <sil>}|
 *
 * where P_{lm}(W) is the language model score, pi_i is the value for the i-th
 * frame in the path leading to W and AM(W) is the (unnormalized) acoustic model
 * score of the transcription W
 */

struct DecoderOptions {
  int beamSize_; // Maximum number of hypothesis we hold after each step
  float beamScore_; // Threshold that keep away hypothesis with score smaller
                    // than best score so far minus beamScore_
  float lmWeight_; // Weight of lm
  float wordScore_; // Score for inserting a word
  float unkScore_; // Score for inserting a unknown word
  bool logAdd_; // If or not use logadd when merging hypothesis
  float silWeight_; // Silence is golden
  ModelType modelType_; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const float beamScore,
      const float lmWeight,
      const float wordScore,
      const float unkScore,
      const bool logAdd,
      const float silWeight,
      const ModelType modelType)
      : beamSize_(beamSize),
        beamScore_(beamScore),
        lmWeight_(lmWeight),
        wordScore_(wordScore),
        unkScore_(unkScore),
        logAdd_(logAdd),
        silWeight_(silWeight),
        modelType_(modelType) {}

  DecoderOptions() {}
};

/**
 * DecoderNode stores information for each hypothesis in the beam.
 */
struct DecoderNode {
  LMStatePtr lmState_; // Language model state
  const TrieNode* lex_; // Trie node in the lexicon
  const DecoderNode* parent_; // Parent hypothesis
  float score_; // Score so far
  int letter_; // Label of letter
  const TrieLabel* label_; // Label of word (-1 if incomplete)
  bool prevBlank_;

  DecoderNode(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const DecoderNode* parent,
      const float score,
      const int letter,
      const TrieLabel* label)
      : lmState_(lmState),
        lex_(lex),
        parent_(parent),
        score_(score),
        letter_(letter),
        label_(label),
        prevBlank_(false) {}

  DecoderNode()
      : lmState_(nullptr),
        lex_(nullptr),
        parent_(nullptr),
        score_(0),
        letter_(-1),
        label_(nullptr) {}

  void updateAsConstructor(
      const LMStatePtr& lmState,
      const TrieNode* lex,
      const DecoderNode* parent,
      const float score,
      const int letter,
      const TrieLabel* label,
      const bool prevBlank) {
    lmState_ = lmState;
    lex_ = lex;
    parent_ = parent;
    score_ = score;
    letter_ = letter;
    label_ = label;
    prevBlank_ = prevBlank;
  }
};

/**
 * Decoder support two typical use cases:
 * Offline manner:
 *  decoder.decode(someData) [returns all hypothesis (transcription)]
 *
 * Online manner:
 *  decoder.decodeBegin() [called only at the beginning of the stream]
 *  while (stream)
 *    decoder.decodeContinue(someData) [one or more calls]
 *    decoder.getBestHypothesis() [returns the best hypothesis (transcription)]
 *    decoder.prune() [prunes the hypothesis space]
 *  decoder.decodeEnd() [called only at the end of the stream]
 *
 * Note: function decoder.prune() deletes hypothesis up until time when called
 * to supports online decoding. It will also add a offset to the scores in beam
 * to avoid underflow/overflow.
 *
 */
class Decoder {
 public:
  Decoder(
      const DecoderOptions& opt,
      const TriePtr lexicon,
      const LMPtr lm,
      const int sil,
      const int blank,
      const TrieLabelPtr unk,
      const std::vector<float>& transitions)
      : opt_(opt),
        lexicon_(lexicon),
        lm_(lm),
        transitions_(transitions),
        sil_(sil),
        blank_(blank),
        unk_(unk),
        nCandidates_(0) {
    candidates_.reserve(kBufferBucketSize);
  }

  void decodeBegin();

  void decodeContinue(const float* emissions, int T, int N);

  void decodeEnd();

  std::tuple<
      std::vector<float>,
      std::vector<std::vector<int>>,
      std::vector<std::vector<int>>>
  decode(const float* emissions, int T, int N);

  int numHypothesis() const;

  int lengthHypothesis() const;

  // Prune the hypothesis space.
  void prune(int lookBack = 0);

  std::tuple<float, std::vector<int>, std::vector<int>> getBestHypothesis(
      int lookBack = 0) const;

 protected:
  DecoderOptions opt_;
  TriePtr lexicon_;
  LMPtr lm_;
  std::vector<float> transitions_;

  std::vector<DecoderNode>
      candidates_; // All the hypothesis candidates (can be larger than
                   // beamsize) for the current frame
  std::vector<DecoderNode*>
      candidatePtrs_; // This vector is only used when sort the candidates_, so
                      // instead of moving around objects, we only need to sort
                      // pointers
  float candidatesBestScore_;
  int sil_; // Index of silence label
  int blank_; // Index of blank label (for CTC)
  TrieLabelPtr unk_; // Trie label for unknown word
  std::unordered_map<int, std::vector<DecoderNode>>
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
      const DecoderNode* parent,
      const float score,
      const int letter,
      const TrieLabel* label,
      const bool prevBlank);

  void candidatesStore(std::vector<DecoderNode>& nextHyp, const bool isSort);

  void mergeNodes(
      DecoderNode* oldNode,
      const DecoderNode* newNode,
      const int logAdd);

  int mergeCandidates(const int size);

  std::tuple<
      std::vector<float>,
      std::vector<std::vector<int>>,
      std::vector<std::vector<int>>>
  storeAllFinalHypothesis() const;

  const DecoderNode* findBestAncestor(int& lookBack) const;
};

} // namespace w2l
