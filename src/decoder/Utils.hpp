/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>
#include <vector>

namespace w2l {

const int kBufferBucketSize = 65536;
const float kNegativeInfinity = -std::numeric_limits<float>::infinity();

enum class ModelType { ASG = 0, CTC = 1 };

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

struct DecodeResult {
  float score_;
  std::vector<int> words_;
  std::vector<int> tokens_;

  DecodeResult() {}

  DecodeResult(int length) {
    words_.resize(length, -1);
    tokens_.resize(length, -1);
  }
};

template <class DecoderState>
void mergeStates(
    DecoderState* oldNode,
    const DecoderState* newNode,
    bool logAdd) {
  float maxScore = std::max(oldNode->score_, newNode->score_);
  if (logAdd) {
    oldNode->score_ = maxScore +
        std::log(std::exp(oldNode->score_ - maxScore) +
                 std::exp(newNode->score_ - maxScore));
  } else {
    oldNode->score_ = maxScore;
  }
}

template <class DecoderState>
int validateCandidates(
    std::vector<DecoderState*>& candidatePtrs,
    std::vector<DecoderState>& candidates,
    const int nCandidates,
    const float bestScore,
    const float beamScore) {
  int nValidHyp = 0;

  for (int i = 0; i < nCandidates; i++) {
    if (candidates[i].score_ >= bestScore - beamScore) {
      if (candidatePtrs.size() == nValidHyp) {
        candidatePtrs.resize(candidatePtrs.size() + kBufferBucketSize);
      }
      candidatePtrs[nValidHyp] = &candidates[i];
      ++nValidHyp;
    }
  }

  return nValidHyp;
}

template <class DecoderState>
void storeTopCandidates(
    std::vector<DecoderState>& nextHyp,
    std::vector<DecoderState*>& candidatePtrs,
    const int nValidHyp,
    const float beamSize,
    const bool returnSorted) {
  auto compareNodesScore = [](const DecoderState* node1,
                              const DecoderState* node2) {
    return node1->score_ > node2->score_;
  };

  if (nValidHyp <= beamSize) {
    if (returnSorted) {
      std::sort(
          candidatePtrs.begin(),
          candidatePtrs.begin() + nValidHyp,
          compareNodesScore);
    }
    nextHyp.resize(nValidHyp);
    for (int i = 0; i < nValidHyp; i++) {
      nextHyp[i] = std::move(*candidatePtrs[i]);
    }
  } else {
    if (!returnSorted) {
      std::nth_element(
          candidatePtrs.begin(),
          candidatePtrs.begin() + beamSize - 1,
          candidatePtrs.begin() + nValidHyp,
          compareNodesScore);
    } else {
      std::partial_sort(
          candidatePtrs.begin(),
          candidatePtrs.begin() + beamSize,
          candidatePtrs.end(),
          compareNodesScore);
    }
    nextHyp.resize(beamSize);
    for (int i = 0; i < beamSize; i++) {
      nextHyp[i] = std::move(*candidatePtrs[i]);
    }
  }
}

template <class DecoderState>
void pruneAndNormalize(
    std::unordered_map<int, std::vector<DecoderState>>& hypothesis,
    const int startFrame,
    const int lookBack) {
  // (1) Move things from back of hypothesis to front.
  for (int i = 0; i <= lookBack; i++) {
    std::swap(hypothesis[i], hypothesis[i + startFrame]);
  }

  // (2) Avoid further back-tracking
  for (DecoderState& hyp : hypothesis[0]) {
    hyp.parent_ = nullptr;
  }

  // (3) For the last frame, subtract the largest score for each hypothesis in
  // it so as to avoid underflow/overflow.
  float largestScore = hypothesis[lookBack].front().score_;
  for (int i = 1; i < hypothesis[lookBack].size(); i++) {
    if (largestScore < hypothesis[lookBack][i].score_) {
      largestScore = hypothesis[lookBack][i].score_;
    }
  }

  for (int i = 0; i < hypothesis[lookBack].size(); i++) {
    hypothesis[lookBack][i].score_ -= largestScore;
  }
}

} // namespace w2l
