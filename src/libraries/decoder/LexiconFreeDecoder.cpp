/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <functional>

#include "libraries/decoder/LexiconFreeDecoder.h"

namespace w2l {

void LexiconFreeDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_.clear();
  candidatePtrs_.clear();
}

void LexiconFreeDecoder::mergeCandidates() {
  auto compareNodesShortList = [&](const LexiconFreeDecoderState* node1,
                                   const LexiconFreeDecoderState* node2) {
    int lmCmp = lm_->compareState(node1->lmState, node2->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else { /* same LmState */
      return node1->score > node2->score;
    }
  };
  std::sort(
      candidatePtrs_.begin(), candidatePtrs_.end(), compareNodesShortList);

  int nHypAfterMerging = 1;
  for (int i = 1; i < candidatePtrs_.size(); i++) {
    if (lm_->compareState(
            candidatePtrs_[i]->lmState,
            candidatePtrs_[nHypAfterMerging - 1]->lmState)) {
      candidatePtrs_[nHypAfterMerging] = candidatePtrs_[i];
      nHypAfterMerging++;
    } else {
      mergeStates(
          candidatePtrs_[nHypAfterMerging - 1], candidatePtrs_[i], opt_.logAdd);
    }
  }

  candidatePtrs_.resize(nHypAfterMerging);
}

void LexiconFreeDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const LexiconFreeDecoderState* parent,
    const double score,
    const int token,
    const bool prevBlank) {
  if (isValidCandidate(candidatesBestScore_, score, opt_.beamThreshold)) {
    candidates_.emplace_back(
        LexiconFreeDecoderState(lmState, parent, score, token, prevBlank));
  }
}

void LexiconFreeDecoder::candidatesStore(
    std::vector<LexiconFreeDecoderState>& nextHyp,
    const bool returnSorted) {
  if (candidates_.empty()) {
    nextHyp.clear();
    return;
  }

  /* Select valid candidates */
  pruneCandidates(
      candidatePtrs_, candidates_, candidatesBestScore_ - opt_.beamThreshold);

  /* Sort by (LmState, lex, score) and copy into next hypothesis */
  mergeCandidates();

  /* Sort hypothesis and select top-K */
  storeTopCandidates(nextHyp, candidatePtrs_, opt_.beamSize, returnSorted);
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
      const LMStatePtr& prevLmState = prevHyp.lmState;

      const int prevIdx = prevHyp.token;
      for (int n = 0; n < N; n++) {
        double score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
          if (prevIdx != sil_) {
            score += opt_.wordScore;
          }
        }

        // DEBUG: CTC branch is not tested
        if ((opt_.criterionType == CriterionType::ASG && n != prevIdx) ||
            (opt_.criterionType == CriterionType::CTC && n != blank_ &&
             (n != prevIdx || prevHyp.prevBlank))) {
          auto lmScoreReturn = lm_->score(prevLmState, n);
          score += lmScoreReturn.second * opt_.lmWeight;

          candidatesAdd(
              lmScoreReturn.first,
              &prevHyp,
              score,
              n,
              false // prevBlank
          );
        } else if (opt_.criterionType == CriterionType::CTC && n == blank_) {
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
    const LMStatePtr& prevLmState = prevHyp.lmState;

    auto lmScoreReturn = lm_->finish(prevLmState);
    candidatesAdd(
        lmScoreReturn.first,
        &prevHyp,
        prevHyp.score + opt_.lmWeight * lmScoreReturn.second,
        sil_,
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
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
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
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
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
