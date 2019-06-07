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

#include "decoder/LexiconFreeDecoder.h"

namespace w2l {

void LexiconFreeDecoder::candidatesReset() {
  nCandidates_ = 0;
  candidatesBestScore_ = kNegativeInfinity;
}

int LexiconFreeDecoder::mergeCandidates(const int size) {
  auto compareNodesShortList = [&](const LexiconFreeDecoderState* node1,
                                   const LexiconFreeDecoderState* node2) {
    int lmCmp = lm_->compareState(node1->lmState_, node2->lmState_);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else { /* same LmState */
      return node1->score_ > node2->score_;
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
            candidatePtrs_[nHypAfterMerging - 1]->lmState_)) {
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

void LexiconFreeDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const LexiconFreeDecoderState* parent,
    const float score,
    const int token,
    const bool prevBlank) {
  if (isGoodCandidate(candidatesBestScore_, score, opt_.beamThreshold_)) {
    if (nCandidates_ == candidates_.size()) {
      candidates_.resize(candidates_.size() + kBufferBucketSize);
    }

    candidates_[nCandidates_] =
        LexiconFreeDecoderState(lmState, parent, score, token, prevBlank);
    ++nCandidates_;
  }
}

void LexiconFreeDecoder::candidatesStore(
    std::vector<LexiconFreeDecoderState>& nextHyp,
    const bool returnSorted) {
  if (nCandidates_ == 0) {
    return;
  }

  /* Select valid candidates */
  int nValidHyp = pruneCandidates(
      candidatePtrs_,
      candidates_,
      nCandidates_,
      candidatesBestScore_,
      opt_.beamThreshold_);

  /* Sort by (LmState, lex, score) and copy into next hypothesis */
  nValidHyp = mergeCandidates(nValidHyp);

  /* Sort hypothesis and select top-K */
  storeTopCandidates(
      nextHyp, candidatePtrs_, nValidHyp, opt_.beamSize_, returnSorted);
}

void LexiconFreeDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconFreeDecoderState>());

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(lm_->start(0), nullptr, 0.0, sil_);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconFreeDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconFreeDecoderState>());
    }
  }

  // Looping over all the frames
  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const LexiconFreeDecoderState& prevHyp : hyp_[startFrame + t]) {
      const LMStatePtr& prevLmState = prevHyp.lmState_;

      const int prevIdx = prevHyp.token_;
      for (int n = 0; n < N; n++) {
        float score = prevHyp.score_ + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType_ == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight_;
          if (prevIdx != sil_) {
            score += opt_.wordScore_;
          }
        }

        // DEBUG: CTC branch is not tested
        if ((opt_.criterionType_ == CriterionType::ASG && n != prevIdx) ||
            (opt_.criterionType_ == CriterionType::CTC && n != blank_ &&
             (n != prevIdx || prevHyp.prevBlank_))) {
          float lmScore = 0;
          LMStatePtr newLmState;
          std::tie(newLmState, lmScore) = lm_->score(prevLmState, n);
          score += lmScore * opt_.lmWeight_;

          candidatesAdd(
              newLmState,
              &prevHyp,
              score,
              n,
              false // prevBlank
          );
        } else if (opt_.criterionType_ == CriterionType::CTC && n == blank_) {
          candidatesAdd(
              prevLmState,
              &prevHyp,
              score,
              n,
              true // prevBlank
          );
        } else {
          candidatesAdd(
              prevLmState,
              &prevHyp,
              score,
              n,
              false // prevBlank
          );
        }
      }
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }
  nDecodedFrames_ += T;
}

void LexiconFreeDecoder::decodeEnd() {
  candidatesReset();
  for (const LexiconFreeDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const LMStatePtr& prevLmState = prevHyp.lmState_;

    float lmScoreEnd;
    LMStatePtr newLmState;
    std::tie(newLmState, lmScoreEnd) = lm_->finish(prevLmState);
    candidatesAdd(
        newLmState,
        &prevHyp,
        prevHyp.score_ + opt_.lmWeight_ * lmScoreEnd,
        -1,
        false // prevBlank
    );
  }

  candidatesStore(hyp_[nDecodedFrames_ - nPrunedFrames_ + 1], true);
  ++nDecodedFrames_;
}

std::vector<DecodeResult> LexiconFreeDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconFreeDecoder::getBestHypothesis(int lookBack) const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  const LexiconFreeDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);

  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconFreeDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconFreeDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconFreeDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  int finalFrame = nDecodedFrames_ - nPrunedFrames_ - lookBack;
  const LexiconFreeDecoderState* bestNode =
      findBestAncestor(hyp_.find(finalFrame)->second, lookBack);
  if (!bestNode) {
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
