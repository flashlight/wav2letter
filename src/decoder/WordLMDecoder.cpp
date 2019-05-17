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

#include "WordLMDecoder.hpp"

namespace w2l {

void WordLMDecoder::candidatesReset() {
  nCandidates_ = 0;
  candidatesBestScore_ = kNegativeInfinity;
}

void WordLMDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const TrieNode* lex,
    const WordLMDecoderState* parent,
    const float score,
    const int token,
    const TrieLabel* word,
    const bool prevBlank) {
  if (score > candidatesBestScore_) {
    candidatesBestScore_ = score;
  }

  if (score >= candidatesBestScore_ - opt_.beamScore_) {
    if (nCandidates_ == candidates_.size()) {
      candidates_.resize(candidates_.size() + kBufferBucketSize);
    }

    candidates_[nCandidates_] =
        WordLMDecoderState(lmState, lex, parent, score, token, word, prevBlank);
    ++nCandidates_;
  }
}

int WordLMDecoder::mergeCandidates(const int size) {
  auto compareNodesShortList = [&](const WordLMDecoderState* node1,
                                   const WordLMDecoderState* node2) {
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
      mergeStates(
          candidatePtrs_[nHypAfterMerging - 1],
          candidatePtrs_[i],
          opt_.logAdd_);
    }
  }

  return nHypAfterMerging;
}

void WordLMDecoder::candidatesStore(
    std::vector<WordLMDecoderState>& nextHyp,
    const bool returnSorted) {
  if (nCandidates_ == 0) {
    return;
  }

  /* Select valid candidates */
  int nValidHyp = validateCandidates(
      candidatePtrs_,
      candidates_,
      nCandidates_,
      candidatesBestScore_,
      opt_.beamScore_);

  /* Sort by (LmState, lex, score) and copy into next hypothesis */
  nValidHyp = mergeCandidates(nValidHyp);

  /* Sort hypothesis and select top-K */
  storeTopCandidates(
      nextHyp, candidatePtrs_, nValidHyp, opt_.beamSize_, returnSorted);
}

void WordLMDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.insert({0, std::vector<WordLMDecoderState>()});

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(
      lm_->start(0), lexicon_->getRoot().get(), nullptr, 0.0, -1, nullptr);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void WordLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.insert({i, std::vector<WordLMDecoderState>()});
    }
  }

  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const WordLMDecoderState& prevHyp : hyp_[startFrame + t]) {
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

void WordLMDecoder::decodeEnd() {
  candidatesReset();
  for (const WordLMDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
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

std::vector<DecodeResult> WordLMDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const std::vector<WordLMDecoderState>& finalHyps =
      hyp_.find(finalFrame)->second;
  int nHyp = finalHyps.size();

  std::vector<DecodeResult> res(nHyp, DecodeResult(finalFrame + 1));

  for (int r = 0; r < nHyp; r++) {
    const WordLMDecoderState* node = &finalHyps[r];
    res[r].score_ = node->score_;
    int i = 0;
    while (node) {
      res[r].words_[finalFrame - i] = (node->word_ ? node->word_->usr_ : -1);
      res[r].tokens_[finalFrame - i] = node->token_;
      node = node->parent_;
      i++;
    }
  }

  return res;
}

DecodeResult WordLMDecoder::getBestHypothesis(int lookBack) const {
  int nHyp = nHypothesis();
  if (nHyp == 0) {
    return DecodeResult();
  }

  int finalFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  DecodeResult res(finalFrame + 1);

  // Search for the best hypothesis
  const WordLMDecoderState* bestNode = findBestAncestor(lookBack);
  if (!bestNode) {
    DecodeResult();
  }
  res.score_ = bestNode->score_;

  // Store the token and word prediction for the best hypothesis
  int i = 0;
  while (bestNode) {
    res.words_[finalFrame - i] = (bestNode->word_ ? bestNode->word_->usr_ : -1);
    res.tokens_[finalFrame - i] = bestNode->lex_->idx_;
    bestNode = bestNode->parent_;
    i++;
  }

  return res;
}

int WordLMDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int WordLMDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

const WordLMDecoderState* WordLMDecoder::findBestAncestor(int& lookBack) const {
  // (1) Find the best hypothesis / best path
  int nHyp = nHypothesis();
  if (nHyp == 0) {
    return nullptr;
  }

  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  const std::vector<WordLMDecoderState>& finalHyps =
      hyp_.find(finalFrame)->second;
  float bestScore = finalHyps.front().score_;
  const WordLMDecoderState* bestNode = finalHyps.data();
  for (int r = 1; r < nHyp; r++) {
    const WordLMDecoderState* node = &finalHyps[r];
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
  while (bestNode && bestNode->parent_) {
    // Check for first emitted word.
    if (bestNode->parent_->word_) {
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

void WordLMDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  const WordLMDecoderState* node = findBestAncestor(lookBack);
  if (!node) {
    return; // Not enough decoded frames to prune
  }

  int startFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  if (startFrame < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (2) Move things from back of hyp_ to front and normalize scores */
  pruneAndNormalize(hyp_, startFrame, lookBack);

  nPrunedFrames_ = nDecodedFrames_ - lookBack;
}

} // namespace w2l
