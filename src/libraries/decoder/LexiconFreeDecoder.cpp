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
#include <numeric>

#include "libraries/decoder/LexiconFreeDecoder.h"

namespace w2l {

void LexiconFreeDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_.clear();
  candidatePtrs_.clear();
}

void LexiconFreeDecoder::mergeCandidates() {
  auto compareNodesShortList = [](const LexiconFreeDecoderState* node1,
                                  const LexiconFreeDecoderState* node2) {
    int lmCmp = node1->lmState->compare(node2->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else if (node1->token != node2->token) {
      return node1->token > node2->token;
    } else if (node1->prevBlank != node2->prevBlank) {
      return node1->prevBlank > node2->prevBlank;
    } else { /* same LmState */
      return node1->score > node2->score;
    }
  };
  std::sort(
      candidatePtrs_.begin(), candidatePtrs_.end(), compareNodesShortList);

  int nHypAfterMerging = 1;
  for (int i = 1; i < candidatePtrs_.size(); i++) {
    if (candidatePtrs_[i]->lmState->compare(
            candidatePtrs_[nHypAfterMerging - 1]->lmState) ||
        candidatePtrs_[i]->token !=
            candidatePtrs_[nHypAfterMerging - 1]->token ||
        candidatePtrs_[i]->prevBlank !=
            candidatePtrs_[nHypAfterMerging - 1]->prevBlank) {
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

  std::vector<size_t> idx(N);
  // Looping over all the frames
  for (int t = 0; t < T; t++) {
    std::iota(idx.begin(), idx.end(), 0);
    if (N > opt_.beamSizeToken) {
      std::partial_sort(
          idx.begin(),
          idx.begin() + opt_.beamSizeToken,
          idx.end(),
          [&t, &N, &emissions](const size_t& l, const size_t& r) {
            return emissions[t * N + l] > emissions[t * N + r];
          });
    }

    candidatesReset();
    for (const LexiconFreeDecoderState& prevHyp : hyp_[startFrame + t]) {
      const int prevIdx = prevHyp.token;

      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        double score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silScore;
        }

        if ((opt_.criterionType == CriterionType::ASG && n != prevIdx) ||
            (opt_.criterionType == CriterionType::CTC && n != blank_ &&
             (n != prevIdx || prevHyp.prevBlank))) {
          auto lmReturn = lm_->score(prevHyp.lmState, n);
          score += lmReturn.second * opt_.lmWeight;

          candidatesAdd(
              lmReturn.first,
              &prevHyp,
              score,
              n,
              false // prevBlank
          );
        } else if (opt_.criterionType == CriterionType::CTC && n == blank_) {
          candidatesAdd(
              prevHyp.lmState,
              &prevHyp,
              score,
              n,
              true // prevBlank
          );
        } else {
          candidatesAdd(
              prevHyp.lmState,
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

    auto lmReturn = lm_->finish(prevLmState);
    candidatesAdd(
        lmReturn.first,
        &prevHyp,
        prevHyp.score + opt_.lmWeight * lmReturn.second,
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
