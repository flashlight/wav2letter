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

#include "libraries/lm/LM.h"

namespace w2l {

const float kNegativeInfinity = -std::numeric_limits<float>::infinity();
const int kLookBackLimit = 100;

enum class CriterionType { ASG = 0, CTC = 1, S2S = 2 };

struct DecoderOptions {
  int beamSize; // Maximum number of hypothesis we hold after each step
  int beamSizeToken; // Maximum number of tokens we consider at each step
  float beamThreshold; // Threshold to prune hypothesis
  float lmWeight; // Weight of lm
  float wordScore; // Score for inserting a word
  float unkScore; // Score for inserting a unknown word
  bool logAdd; // If or not use logadd when merging hypothesis
  float silWeight; // Silence is golden
  CriterionType criterionType; // CTC or ASG

  DecoderOptions(
      const int beamSize,
      const int beamSizeToken,
      const float beamThreshold,
      const float lmWeight,
      const float wordScore,
      const float unkScore,
      const bool logAdd,
      const float silWeight,
      const CriterionType criterionType)
      : beamSize(beamSize),
        beamSizeToken(beamSizeToken),
        beamThreshold(beamThreshold),
        lmWeight(lmWeight),
        wordScore(wordScore),
        unkScore(unkScore),
        logAdd(logAdd),
        silWeight(silWeight),
        criterionType(criterionType) {}

  DecoderOptions() {}
};

struct DecodeResult {
  double score;
  std::vector<int> words;
  std::vector<int> tokens;

  explicit DecodeResult(int length = 0)
      : score(0), words(length, -1), tokens(length, -1) {}
};

template <class DecoderState>
void mergeStates(
    DecoderState* oldNode,
    const DecoderState* newNode,
    bool logAdd) {
  double maxScore = std::max(oldNode->score, newNode->score);
  if (logAdd) {
    double minScore = std::min(oldNode->score, newNode->score);
    oldNode->score = maxScore + std::log1p(std::exp(minScore - maxScore));
  } else {
    oldNode->score = maxScore;
  }
}

bool isValidCandidate(
    double& bestScore,
    const double score,
    const double beamThreshold);

template <class DecoderState>
void pruneCandidates(
    std::vector<DecoderState*>& candidatePtrs,
    std::vector<DecoderState>& candidates,
    const float threshold) {
  for (auto& candidate : candidates) {
    if (candidate.score >= threshold) {
      candidatePtrs.emplace_back(&candidate);
    }
  }
}

template <class DecoderState>
void storeTopCandidates(
    std::vector<DecoderState>& nextHyp,
    std::vector<DecoderState*>& candidatePtrs,
    const int beamSize,
    const bool returnSorted) {
  auto compareNodes = [](const DecoderState* node1, const DecoderState* node2) {
    return node1->score > node2->score;
  };

  int nValidHyp = candidatePtrs.size();
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
  res.score = node_->score;

  int i = 0;
  while (node_) {
    res.words[finalFrame - i] = node_->getWord();
    res.tokens[finalFrame - i] = node_->token;
    node_ = node_->parent;
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

  double bestScore = finalHyps.front().score;
  const DecoderState* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const DecoderState* node = &finalHyps[r];
    if (node->score > bestScore) {
      bestScore = node->score;
      bestNode = node;
    }
  }

  int n = 0;
  while (bestNode && n < lookBack) {
    n++;
    bestNode = bestNode->parent;
  }

  const int maxLookBack = lookBack + kLookBackLimit;
  while (bestNode) {
    // Check for first emitted word.
    if (bestNode->isComplete()) {
      break;
    }

    n++;
    bestNode = bestNode->parent;

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
    hyp.parent = nullptr;
  }

  // (3) For each hypothesis in the last frame, subtract the largest score so as
  // to avoid underflow/overflow.
  double largestScore = hypothesis[lookBack].front().score;
  for (int i = 1; i < hypothesis[lookBack].size(); i++) {
    if (largestScore < hypothesis[lookBack][i].score) {
      largestScore = hypothesis[lookBack][i].score;
    }
  }

  for (int i = 0; i < hypothesis[lookBack].size(); i++) {
    hypothesis[lookBack][i].score -= largestScore;
  }
}

template <class DecoderState>
void updateLMCache(const LMPtr& lm, std::vector<DecoderState>& hypothesis) {
  // For ConvLM update cache
  std::vector<LMStatePtr> states;
  for (const auto& hyp : hypothesis) {
    states.emplace_back(hyp.lmState);
  }
  lm->updateCache(states);
}

} // namespace w2l
