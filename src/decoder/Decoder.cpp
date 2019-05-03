/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <unordered_map>

#include "Decoder.hpp"

namespace w2l {

void Decoder::candidatesReset() {
  nCandidates_ = 0;
  candidatesBestScore_ = kNegativeInfinity;
}

void Decoder::candidatesAdd(
    const LMStatePtr& lmState,
    const TrieNode* lex,
    const DecoderNode* parent,
    const float score,
    const int letter,
    const TrieLabel* label,
    const bool prevBlank) {
  if (score > candidatesBestScore_) {
    candidatesBestScore_ = score;
  }

  if (score >= candidatesBestScore_ - opt_.beamScore_) {
    if (nCandidates_ == candidates_.size()) {
      candidates_.resize(candidates_.size() + kBufferBucketSize);
    }

    candidates_[nCandidates_].updateAsConstructor(
        lmState, lex, parent, score, letter, label, prevBlank);
    ++nCandidates_;
  }
}

void Decoder::mergeNodes(
    DecoderNode* oldNode,
    const DecoderNode* newNode,
    int logAdd) {
  float maxScore = std::max(oldNode->score_, newNode->score_);
  if (logAdd) {
    oldNode->score_ = maxScore +
        std::log(std::exp(oldNode->score_ - maxScore) +
                 std::exp(newNode->score_ - maxScore));
  } else {
    oldNode->score_ = maxScore;
  }
}

int Decoder::mergeCandidates(const int size) {
  auto compareNodesShortList = [&](const DecoderNode* node1,
                                   const DecoderNode* node2) {
    int lmCmp = lm_->compareState(node1->lmState_, node2->lmState_);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else { /* same LmState */
      if (node1->lex_ != node2->lex_) {
        return node1->lex_ > node2->lex_;
      } else { /* same LmState, same lex */
        return node1->score_ > node2->score_;
      }
    }
  };
  std::sort(
      candidatePtrs_.begin(),
      candidatePtrs_.begin() + size,
      compareNodesShortList);

  int nHypAfterMerging = 1;
  for (int i = 1; i < size; i++) {
    if (lm_->compareState(
            candidatePtrs_[i]->lmState_,
            candidatePtrs_[nHypAfterMerging - 1]->lmState_) ||
        candidatePtrs_[i]->lex_ != candidatePtrs_[nHypAfterMerging - 1]->lex_) {
      candidatePtrs_[nHypAfterMerging] = candidatePtrs_[i];
      nHypAfterMerging++;
    } else {
      mergeNodes(
          candidatePtrs_[nHypAfterMerging - 1],
          candidatePtrs_[i],
          opt_.logAdd_);
    }
  }

  return nHypAfterMerging;
}

void Decoder::candidatesStore(
    std::vector<DecoderNode>& nextHyp,
    const bool isSort) {
  if (nCandidates_ == 0) {
    return;
  }
  int nValidHyp = 0;

  /* Select valid candidates */
  for (int i = 0; i < nCandidates_; i++) {
    if (candidates_[i].score_ >= candidatesBestScore_ - opt_.beamScore_) {
      if (candidatePtrs_.size() == nValidHyp) {
        candidatePtrs_.resize(candidatePtrs_.size() + kBufferBucketSize);
      }
      candidatePtrs_[nValidHyp] = &candidates_[i];
      ++nValidHyp;
    }
  }

  /* Sort by (LmState, lex, score) and copy into next hypothesis */
  nValidHyp = mergeCandidates(nValidHyp);

  /* Sort hypothesis and select top-K */
  auto compareNodesScore = [](const DecoderNode* node1,
                              const DecoderNode* node2) {
    return node1->score_ > node2->score_;
  };

  if (nValidHyp <= opt_.beamSize_) {
    if (isSort) {
      std::sort(
          candidatePtrs_.begin(),
          candidatePtrs_.begin() + nValidHyp,
          compareNodesScore);
    }
    nextHyp.resize(nValidHyp);
    for (int i = 0; i < nValidHyp; i++) {
      nextHyp[i] = std::move(*candidatePtrs_[i]);
    }
  } else {
    if (!isSort) {
      std::nth_element(
          candidatePtrs_.begin(),
          candidatePtrs_.begin() + opt_.beamSize_ - 1,
          candidatePtrs_.begin() + nValidHyp,
          compareNodesScore);
    } else {
      std::sort(
          candidatePtrs_.begin(),
          candidatePtrs_.begin() + nValidHyp,
          compareNodesScore);
    }
    nextHyp.resize(opt_.beamSize_);
    for (int i = 0; i < opt_.beamSize_; i++) {
      nextHyp[i] = std::move(*candidatePtrs_[i]);
    }
  }
}

void Decoder::decodeBegin() {
  hyp_.clear();
  hyp_.insert({0, std::vector<DecoderNode>()});

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(
      lm_->start(0), lexicon_->getRoot().get(), nullptr, 0.0, -1, nullptr);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void Decoder::decodeContinue(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.insert({i, std::vector<DecoderNode>()});
    }
  }

  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const DecoderNode& prevHyp : hyp_[startFrame + t]) {
      const TrieNode* prevLex = prevHyp.lex_;
      const int prevIdx = prevLex->idx_;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot().get() ? 0 : prevLex->maxScore_;
      const LMStatePtr& prevLmState = prevHyp.lmState_;

      /* (1) Try children */
      for (auto& child : prevLex->children_) {
        int n = child.first;
        const TrieNodePtr& lex = child.second;
        float score = prevHyp.score_ + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 && opt_.modelType_ == ModelType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight_;
        }

        // We eat-up a new token
        if (opt_.modelType_ != ModelType::CTC || prevHyp.prevBlank_ ||
            n != prevIdx) {
          if (!lex->children_.empty()) {
            candidatesAdd(
                prevLmState,
                lex.get(),
                &prevHyp,
                score + opt_.lmWeight_ * (lex->maxScore_ - lexMaxScore),
                n,
                nullptr,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (int i = 0; i < lex->nLabel_; i++) {
          float lmScore;
          const LMStatePtr newLmState =
              lm_->score(prevLmState, lex->label_[i]->lm_, lmScore);
          candidatesAdd(
              newLmState,
              lexicon_->getRoot().get(),
              &prevHyp,
              score + opt_.lmWeight_ * (lmScore - lexMaxScore) +
                  opt_.wordScore_,
              n,
              lex->label_[i].get(),
              false // prevBlank
          );
        }

        // If we got an unknown word
        if (lex->nLabel_ == 0 && (opt_.unkScore_ > kNegativeInfinity)) {
          float lmScore;
          const LMStatePtr newLmState =
              lm_->score(prevLmState, unk_->lm_, lmScore);
          candidatesAdd(
              newLmState,
              lexicon_->getRoot().get(),
              &prevHyp,
              score + opt_.lmWeight_ * (lmScore - lexMaxScore) + opt_.unkScore_,
              n,
              unk_.get(),
              false // prevBlank
          );
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.modelType_ != ModelType::CTC || !prevHyp.prevBlank_) {
        int n = prevIdx;
        float score = prevHyp.score_ + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 && opt_.modelType_ == ModelType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight_;
        }

        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            nullptr,
            false // prevBlank
        );
      }

      /* (3) CTC only, try blank */
      if (opt_.modelType_ == ModelType::CTC) {
        int n = blank_;
        float score = prevHyp.score_ + emissions[t * N + n];
        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            nullptr,
            true // prevBlank
        );
      }
      // finish proposing
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
  }
  nDecodedFrames_ += T;
}

void Decoder::decodeEnd() {
  candidatesReset();
  for (const DecoderNode& prevHyp : hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const TrieNode* prevLex = prevHyp.lex_;
    const LMStatePtr& prevLmState = prevHyp.lmState_;

    float lmScoreEnd;
    LMStatePtr newLmState = lm_->finish(prevLmState, lmScoreEnd);
    candidatesAdd(
        newLmState,
        prevLex,
        &prevHyp,
        prevHyp.score_ + opt_.lmWeight_ * lmScoreEnd,
        -1,
        nullptr,
        false // prevBlank
    );
  }

  candidatesStore(hyp_[nDecodedFrames_ - nPrunedFrames_ + 1], true);
  ++nDecodedFrames_;
}

std::tuple<
    std::vector<float>,
    std::vector<std::vector<int>>,
    std::vector<std::vector<int>>>
Decoder::decode(const float* emissions, int T, int N) {
  decodeBegin();
  decodeContinue(emissions, T, N);
  decodeEnd();
  return storeAllFinalHypothesis();
}

std::tuple<
    std::vector<float>,
    std::vector<std::vector<int>>,
    std::vector<std::vector<int>>>
Decoder::storeAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const std::vector<DecoderNode>& finalHyps = hyp_.find(finalFrame)->second;
  int nHyp = finalHyps.size();

  std::vector<float> scores(nHyp);
  std::vector<std::vector<int>> wordPredictions(
      nHyp, std::vector<int>(finalFrame + 1, -1));
  std::vector<std::vector<int>> letterPredictions(
      nHyp, std::vector<int>(finalFrame + 1, -1));

  for (int r = 0; r < nHyp; r++) {
    const DecoderNode* node = &finalHyps[r];
    scores[r] = node->score_;
    int i = 0;
    while (node) {
      wordPredictions[r][finalFrame - i] =
          (node->label_ ? node->label_->usr_ : -1);
      letterPredictions[r][finalFrame - i] = node->letter_;
      node = node->parent_;
      i++;
    }
  }

  return std::tie(scores, wordPredictions, letterPredictions);
}

std::tuple<float, std::vector<int>, std::vector<int>>
Decoder::getBestHypothesis(int lookBack) const {
  int nHyp = numHypothesis();
  if (nHyp == 0) {
    return std::make_tuple(0.0, std::vector<int>{}, std::vector<int>{});
  }

  // Search for the best hypothesis
  const DecoderNode* bestNode = findBestAncestor(lookBack);
  if (!bestNode) {
    return std::make_tuple(0.0, std::vector<int>{}, std::vector<int>{});
  }
  float score = bestNode->score_;

  // Store the letter and word prediction for the best hypothesis
  int finalFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  std::vector<int> wordPrediction(finalFrame + 1, -1);
  std::vector<int> letterPrediction(finalFrame + 1, -1);
  int i = 0;
  while (bestNode) {
    wordPrediction[finalFrame - i] =
        (bestNode->label_ ? bestNode->label_->usr_ : -1);
    letterPrediction[finalFrame - i] = bestNode->lex_->idx_;
    bestNode = bestNode->parent_;
    i++;
  }

  return std::make_tuple(score, wordPrediction, letterPrediction);
}

int Decoder::numHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int Decoder::lengthHypothesis() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

const DecoderNode* Decoder::findBestAncestor(int& lookBack) const {
  // (1) Find the best hypothesis / best path
  int nHyp = numHypothesis();
  if (nHyp == 0) {
    return nullptr;
  }

  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const std::vector<DecoderNode>& finalHyps = hyp_.find(finalFrame)->second;
  float bestScore = finalHyps.front().score_;
  const DecoderNode* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const DecoderNode* node = &finalHyps[r];
    if (node->score_ > bestScore) {
      bestScore = node->score_;
      bestNode = node;
    }
  }

  // (2) Search for the last emitted word in the best path
  int n = 0;
  while (bestNode && n < lookBack) {
    n++;
    bestNode = bestNode->parent_;
  }

  const int maxLookBack = lookBack + 100;
  while (bestNode) {
    // Check for first emitted word.
    if (bestNode->label_) {
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

void Decoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  const DecoderNode* node = findBestAncestor(lookBack);
  if (!node) {
    return; // Not enough decoded frames to prune
  }
  const LMStatePtr bestLmState = node->lmState_;

  /* (2) Move things from back of hyp_ to front. */
  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  for (int i = 0; i <= lookBack; i++) {
    std::swap(hyp_[i], hyp_[i + startFrame]);
  }

  for (DecoderNode& hyp : hyp_[0]) {
    hyp.parent_ = nullptr;
    hyp.label_ = nullptr;
  }

  nPrunedFrames_ = nDecodedFrames_ - lookBack;

  // (3) For the last frame hyp_[lookBack], subtract the largest
  // score for each hypothesis in it so as to avoid
  // underflow/overflow.
  float largestScore = hyp_[lookBack].front().score_;
  for (int i = 1; i < hyp_[lookBack].size(); i++) {
    if (largestScore < hyp_[lookBack][i].score_) {
      largestScore = hyp_[lookBack][i].score_;
    }
  }

  for (int i = 0; i < hyp_[lookBack].size(); i++) {
    hyp_[lookBack][i].score_ -= largestScore;
  }
}

} // namespace w2l
