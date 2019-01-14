/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "LM.hpp"
#include "Trie.hpp"

namespace w2l {

const int kBufferBucketSize = 65536;

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
  bool forceEndSil_; // If or not force ending in a sil
  bool logAdd_; // If or not use logadd when merging hypothesis
  float silWeight_; // Silence is golden
  ModelType modelType_; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const float beamScore,
      const float lmWeight,
      const float wordScore,
      const float unkScore,
      const bool forceEndSil,
      const bool logAdd,
      const float silWeight,
      const ModelType modelType)
      : beamSize_(beamSize),
        beamScore_(beamScore),
        lmWeight_(lmWeight),
        wordScore_(wordScore),
        unkScore_(unkScore),
        forceEndSil_(forceEndSil),
        logAdd_(logAdd),
        silWeight_(silWeight),
        modelType_(modelType) {}
};

/**
 * DecoderNode stores information for each hypothesis in the beam.
 */
struct DecoderNode {
  LMStatePtr lmState_; // Language model state
  TrieNodePtr lex_; // Trie node in the lexicon
  const DecoderNode* parent_; // Parent hypothesis
  float score_; // Score so far
  TrieLabelPtr label_; // Label of word (-1 if incomplete)
  bool prevBlank_;

  DecoderNode(
      const LMStatePtr& lmState,
      const TrieNodePtr& lex,
      const DecoderNode* parent,
      const float score,
      const TrieLabelPtr& label)
      : lmState_(lmState),
        lex_(lex),
        parent_(parent),
        score_(score),
        label_(label),
        prevBlank_(false) {}

  DecoderNode()
      : lmState_(nullptr),
        lex_(nullptr),
        parent_(nullptr),
        score_(0),
        label_(nullptr) {}

  void updateAsConstructor(
      const LMStatePtr& lmState,
      const TrieNodePtr& lex,
      const DecoderNode* parent,
      const float score,
      const TrieLabelPtr& label,
      const bool prevBlank) {
    lmState_ = lmState;
    lex_ = lex;
    parent_ = parent;
    score_ = score;
    label_ = label;
    prevBlank_ = prevBlank;
  }
};

class Decoder {
 public:
  Decoder(
      const TriePtr lexicon,
      const LMPtr lm,
      const int sil,
      const int blank,
      const TrieLabelPtr unk)
      : lexicon_(lexicon),
        lm_(lm),
        sil_(sil),
        blank_(blank),
        unk_(unk),
        nCandidates_(0) {
    candidates_.reserve(kBufferBucketSize);
  }

  std::tuple<
      std::vector<float>,
      std::vector<std::vector<int>>,
      std::vector<std::vector<int>>>
  decode(
      const DecoderOptions& opt,
      const float* transitions,
      const float* emissions,
      int T,
      int N);

 private:
  TriePtr lexicon_;
  LMPtr lm_;
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
  std::vector<std::vector<DecoderNode>>
      hyp_; // Vector of hypothesis for all the frames so far
  int nCandidates_; // Total number of candidates in candidates_. Note that
                    // candidates is not always equal to candidates_.size()
                    // since we do not refresh the buffer for candidates_ in
                    // memory through out the whole decoding process.

  std::tuple<
      std::vector<float>,
      std::vector<std::vector<int>>,
      std::vector<std::vector<int>>>
  storeHypothesis(int T) const;

  void candidatesReset();

  void candidatesAdd(
      const LMStatePtr& lmState,
      const TrieNodePtr& lex,
      const DecoderNode* parent,
      const float score,
      const float beamScore,
      const TrieLabelPtr& label,
      const bool prevBlank);

  void candidatesStore(
      const DecoderOptions& opt,
      std::vector<DecoderNode>& nextHyp,
      const bool isSort);

  void mergeNodes(DecoderNode* oldNode, const DecoderNode* newNode, int logAdd);
};

} // namespace w2l
