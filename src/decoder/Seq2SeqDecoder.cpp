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
#include <iostream>

#include "decoder/Seq2SeqDecoder.h"

namespace w2l {

void Seq2SeqDecoder::candidatesReset() {
  nCandidates_ = 0;
  candidatesBestScore_ = kNegativeInfinity;
}

int Seq2SeqDecoder::mergeCandidates(const int size) {
  auto compareNodesShortList = [&](const Seq2SeqDecoderState* node1,
                                   const Seq2SeqDecoderState* node2) {
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

void Seq2SeqDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const Seq2SeqDecoderState* parent,
    const float score,
    const int token,
    const AMStatePtr& amState) {
  if (isGoodCandidate(candidatesBestScore_, score, opt_.beamThreshold_)) {
    if (nCandidates_ == candidates_.size()) {
      candidates_.resize(candidates_.size() + kBufferBucketSize);
    }

    candidates_[nCandidates_] =
        Seq2SeqDecoderState(lmState, parent, score, token, amState);
    ++nCandidates_;
  }
}

void Seq2SeqDecoder::candidatesStore(
    std::vector<Seq2SeqDecoderState>& nextHyp,
    const bool isSort) {
  if (nCandidates_ == 0) {
    nextHyp.clear();
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
      nextHyp, candidatePtrs_, nValidHyp, opt_.beamSize_, isSort);
}

void Seq2SeqDecoder::decodeStep(const float* emissions, int T, int N) {
  // Extend hyp_ buffer
  if (hyp_.size() < maxOutputLength_ + 2) {
    for (int i = hyp_.size(); i < maxOutputLength_ + 2; i++) {
      hyp_.emplace(i, std::vector<Seq2SeqDecoderState>());
    }
  }

  // Start from here.
  hyp_[0].clear();
  hyp_[0].emplace_back(lm_->start(0), nullptr, 0.0, -1, nullptr);

  auto compare = [](const Seq2SeqDecoderState& n1,
                    const Seq2SeqDecoderState& n2) {
    return n1.score_ > n2.score_;
  };
  completedCandidates_.resize(0);

  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset();

    // Batch forwarding
    rawY.resize(0);
    rawPrevStates.resize(0);
    if (hyp_[t].empty()) {
      t--;
      break;
    }
    for (const Seq2SeqDecoderState& prevHyp : hyp_[t]) {
      const AMStatePtr& prevState = prevHyp.amState_;
      if (prevHyp.token_ == eos_) {
        completedCandidates_.push_back(prevHyp);
        continue;
      }
      rawY.push_back(prevHyp.token_);
      rawPrevStates.push_back(prevState);
    }
    if (rawY.size() == 0) {
      if (completedCandidates_.size() >= opt_.beamSize_) {
        std::partial_sort(
            completedCandidates_.begin(),
            completedCandidates_.begin() + opt_.beamSize_,
            completedCandidates_.end(),
            compare);
        completedCandidates_.resize(opt_.beamSize_);
      }
      break;
    }

    std::vector<std::vector<float>> amScores;
    std::vector<AMStatePtr> outStates;

    std::tie(amScores, outStates) =
        amUpdateFunc_(emissions, N, T, rawY, rawPrevStates, t);

    // Generate new hypothesis
    for (int hypo = 0, validHypo = 0; hypo < hyp_[t].size(); hypo++) {
      const Seq2SeqDecoderState& prevHyp = hyp_[t][hypo];
      // Move away completed hypothesis
      if (prevHyp.token_ == eos_) {
        continue;
      }

      const AMStatePtr& outState = outStates[validHypo];
      if (!outState) {
        validHypo++;
        continue;
      }

      auto prevLmState = prevHyp.lmState_;

      const float maxAmScore = *std::max_element(
          amScores[validHypo].begin(), amScores[validHypo].end());

      for (int n = 0; n < amScores[validHypo].size(); n++) {
        float score = prevHyp.score_ + amScores[validHypo][n];
        float lmScore;
        LMStatePtr newLmState;

        /* (1) Try eos */
        if (n == eos_ &&
            amScores[validHypo][eos_] >= hardSelection_ * maxAmScore) {
          std::tie(newLmState, lmScore) = lm_->finish(prevLmState);

          candidatesAdd(
              newLmState,
              &prevHyp,
              score + opt_.lmWeight_ * lmScore,
              n,
              nullptr);
        }

        /* (2) Try normal token */
        if (n != eos_ &&
            amScores[validHypo][n] >= maxAmScore - softSelection_) {
          std::tie(newLmState, lmScore) = lm_->score(prevLmState, n);
          candidatesAdd(
              newLmState,
              &prevHyp,
              score + opt_.wordScore_ + opt_.lmWeight_ * lmScore,
              n,
              outState);
        }
      }
      validHypo++;
    }
    candidatesStore(hyp_[t + 1], true);
    updateLMCache(lm_, hyp_[t + 1]);

    // Sort completed candidates if necessary
    if (completedCandidates_.size() >= opt_.beamSize_) {
      std::partial_sort(
          completedCandidates_.begin(),
          completedCandidates_.begin() + opt_.beamSize_,
          completedCandidates_.end(),
          compare);
      completedCandidates_.resize(opt_.beamSize_);
    }
  } // End of decoding

  if (completedCandidates_.size() < opt_.beamSize_) {
    std::sort(
        completedCandidates_.begin(), completedCandidates_.end(), compare);
  }

  if (completedCandidates_.size() > 0) {
    hyp_[maxOutputLength_ + 1].resize(completedCandidates_.size());
    for (int i = 0; i < completedCandidates_.size(); i++) {
      hyp_[maxOutputLength_ + 1][i] = std::move(completedCandidates_[i]);
    }
  } else {
    std::cout << "[WARNING] No completed candidates.\n";
    hyp_[maxOutputLength_ + 1].resize(hyp_[t].size());
    for (int i = 0; i < hyp_[t].size(); i++) {
      hyp_[maxOutputLength_ + 1][i] = std::move(hyp_[t][i]);
    }
  }
}

std::vector<DecodeResult> Seq2SeqDecoder::getAllFinalHypothesis() const {
  return getAllHypothesis(hyp_.find(maxOutputLength_ + 1)->second, hyp_.size());
}

DecodeResult Seq2SeqDecoder::getBestHypothesis(int /* unused */) const {
  return getHypothesis(
      hyp_.find(maxOutputLength_ + 1)->second.data(), hyp_.size());
}

void Seq2SeqDecoder::prune(int /* unused */) {
  return;
}

} // namespace w2l
