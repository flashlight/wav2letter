/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "decoder/LM.h"

namespace w2l {

const int kBufferBucketSize = 65536;
const float kNegativeInfinity = -std::numeric_limits<float>::infinity();
const int kLookBackLimit = 100;

enum class CriterionType { ASG = 0, CTC = 1, S2S = 2 };

struct DecoderOptions {
  int beamSize_; // Maximum number of hypothesis we hold after each step
  float beamThreshold_; // Threshold to prune hypothesis
  float lmWeight_; // Weight of lm
  float wordScore_; // Score for inserting a word
  float unkScore_; // Score for inserting a unknown word
  bool logAdd_; // If or not use logadd when merging hypothesis
  float silWeight_; // Silence is golden
  CriterionType criterionType_; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const float beamThreshold,
      const float lmWeight,
      const float wordScore,
      const float unkScore,
      const bool logAdd,
      const float silWeight,
      const CriterionType CriterionType)
      : beamSize_(beamSize),
        beamThreshold_(beamThreshold),
        lmWeight_(lmWeight),
        wordScore_(wordScore),
        unkScore_(unkScore),
        logAdd_(logAdd),
        silWeight_(silWeight),
        criterionType_(CriterionType) {}

  DecoderOptions() {}
};

struct DecodeResult {
  float score_;
  std::vector<int> words_;
  std::vector<int> tokens_;

  explicit DecodeResult(int length = 0)
      : score_(0), words_(length, -1), tokens_(length, -1) {}
};

template <class DecoderState>
void mergeStates(
    DecoderState* oldNode,
    const DecoderState* newNode,
    bool logAdd) {
  float maxScore = std::max(oldNode->score_, newNode->score_);
  if (logAdd) {
    float minScore = std::min(oldNode->score_, newNode->score_);
    oldNode->score_ = maxScore + std::log1p(std::exp(minScore - maxScore));
  } else {
    oldNode->score_ = maxScore;
  }
}

bool isGoodCandidate(
    float& bestScore,
    const float score,
    const float beamThreshold);

template <class DecoderState>
int pruneCandidates(
    std::vector<DecoderState*>& candidatePtrs,
    std::vector<DecoderState>& candidates,
    const int nCandidates,
    const float bestScore,
    const float beamThreshold) {
  int nValidHyp = 0;

  for (int i = 0; i < nCandidates; i++) {
    if (candidates[i].score_ >= bestScore - beamThreshold) {
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
    int nValidHyp,
    const int beamSize,
    const bool returnSorted) {
  auto compareNodes = [](const DecoderState* node1, const DecoderState* node2) {
    return node1->score_ > node2->score_;
  };

  int finalSize = std::min(nValidHyp, beamSize);
  if (!returnSorted && nValidHyp > beamSize) {
    std::nth_element(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodes);
  } else if (returnSorted) {
    std::partial_sort(
        candidatePtrs.begin(),
        candidatePtrs.begin() + finalSize,
        candidatePtrs.begin() + nValidHyp,
        compareNodes);
  }

  nextHyp.resize(finalSize);
  for (int i = 0; i < finalSize; i++) {
    nextHyp[i] = std::move(*candidatePtrs[i]);
  }
}

template <class DecoderState>
DecodeResult getHypothesis(const DecoderState* node, const int finalFrame) {
  const DecoderState* node_ = node;
  if (!node_) {
    return DecodeResult();
  }

  DecodeResult res(finalFrame + 1);
  res.score_ = node_->score_;

  int i = 0;
  while (node_) {
    res.words_[finalFrame - i] = node_->getWord();
    res.tokens_[finalFrame - i] = node_->token_;
    node_ = node_->parent_;
    i++;
  }

  return res;
}

template <class DecoderState>
std::vector<DecodeResult> getAllHypothesis(
    const std::vector<DecoderState>& finalHyps,
    const int finalFrame) {
  int nHyp = finalHyps.size();

  std::vector<DecodeResult> res(nHyp);

  for (int r = 0; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    res[r] = getHypothesis(node, finalFrame);
  }

  return res;
}

template <class DecoderState>
const DecoderState* findBestAncestor(
    const std::vector<DecoderState>& finalHyps,
    int& lookBack) {
  int nHyp = finalHyps.size();
  if (nHyp == 0) {
    return nullptr;
  }

  float bestScore = finalHyps.front().score_;
  const DecoderState* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    if (node->score_ > bestScore) {
      bestScore = node->score_;
      bestNode = node;
    }
  }

  int n = 0;
  while (bestNode && n < lookBack) {
    n++;
    bestNode = bestNode->parent_;
  }

  const int maxLookBack = lookBack + kLookBackLimit;
  while (bestNode) {
    // Check for first emitted word.
    if (bestNode->isComplete()) {
      break;
    }

    n++;
    bestNode = bestNode->parent_;

    if (n == maxLookBack) {
      break;
    }
  }

  lookBack = n;
  return bestNode;
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

  // (3) For each hypothesis in the last frame, subtract the largest score so as
  // to avoid underflow/overflow.
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

template <class DecoderState>
void updateLMCache(const LMPtr& lm, std::vector<DecoderState>& hypothesis) {
  // For ConvLM update cache
  std::vector<LMStatePtr> states;
  for (const auto& hyp : hypothesis) {
    states.emplace_back(hyp.lmState_);
  }
  lm->updateCache(states);
}

} // namespace w2l
