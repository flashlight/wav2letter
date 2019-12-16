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

#include "libraries/decoder/LexiconFreeSeq2SeqDecoder.h"

namespace w2l {

void LexiconFreeSeq2SeqDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_.clear();
  candidatePtrs_.clear();
}

void LexiconFreeSeq2SeqDecoder::mergeCandidates() {
  auto compareNodesShortList = [](const LexiconFreeSeq2SeqDecoderState* node1,
                                  const LexiconFreeSeq2SeqDecoderState* node2) {
    int lmCmp = node1->lmState->compare(node2->lmState);
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
    if (candidatePtrs_[i]->lmState->compare(
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

void LexiconFreeSeq2SeqDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const LexiconFreeSeq2SeqDecoderState* parent,
    const double score,
    const int token,
    const AMStatePtr& amState) {
  if (isValidCandidate(candidatesBestScore_, score, opt_.beamThreshold)) {
    candidates_.emplace_back(
        LexiconFreeSeq2SeqDecoderState(lmState, parent, score, token, amState));
  }
}

void LexiconFreeSeq2SeqDecoder::candidatesStore(
    std::vector<LexiconFreeSeq2SeqDecoderState>& nextHyp,
    const bool isSort) {
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
  storeTopCandidates(nextHyp, candidatePtrs_, opt_.beamSize, isSort);
}

void LexiconFreeSeq2SeqDecoder::decodeStep(
    const float* emissions,
    int T,
    int N) {
  // Extend hyp_ buffer
  if (hyp_.size() < maxOutputLength_ + 2) {
    for (int i = hyp_.size(); i < maxOutputLength_ + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconFreeSeq2SeqDecoderState>());
    }
  }

  // Start from here.
  hyp_[0].clear();
  hyp_[0].emplace_back(lm_->start(0), nullptr, 0.0, -1, nullptr);

  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset();

    // Batch forwarding
    rawY_.clear();
    rawPrevStates_.clear();
    for (const LexiconFreeSeq2SeqDecoderState& prevHyp : hyp_[t]) {
      const AMStatePtr& prevState = prevHyp.amState;
      if (prevHyp.token == eos_) {
        continue;
      }
      rawY_.push_back(prevHyp.token);
      rawPrevStates_.push_back(prevState);
    }
    if (rawY_.size() == 0) {
      break;
    }

    std::vector<std::vector<float>> amScores;
    std::vector<AMStatePtr> outStates;

    std::tie(amScores, outStates) =
        amUpdateFunc_(emissions, N, T, rawY_, rawPrevStates_, t);

    std::vector<size_t> idx(amScores.back().size());

    // Generate new hypothesis
    for (int hypo = 0, validHypo = 0; hypo < hyp_[t].size(); hypo++) {
      const LexiconFreeSeq2SeqDecoderState& prevHyp = hyp_[t][hypo];
      // Change nothing for completed hypothesis
      if (prevHyp.token == eos_) {
        candidatesAdd(prevHyp.lmState, &prevHyp, prevHyp.score, eos_, nullptr);
        continue;
      }

      const AMStatePtr& outState = outStates[validHypo];
      if (!outState) {
        validHypo++;
        continue;
      }

      std::iota(idx.begin(), idx.end(), 0);
      if (amScores[validHypo].size() > opt_.beamSizeToken) {
        std::partial_sort(
            idx.begin(),
            idx.begin() + opt_.beamSizeToken,
            idx.end(),
            [&amScores, &validHypo](const size_t& l, const size_t& r) {
              return amScores[validHypo][l] > amScores[validHypo][r];
            });
      }

      for (int r = 0;
           r < std::min(amScores[validHypo].size(), (size_t)opt_.beamSizeToken);
           r++) {
        int n = idx[r];
        double score = prevHyp.score + amScores[validHypo][n];

        if (n == eos_) { /* (1) Try eos */
          auto lmScoreReturn = lm_->finish(prevHyp.lmState);

          candidatesAdd(
              lmScoreReturn.first,
              &prevHyp,
              score + opt_.eosScore + opt_.lmWeight * lmScoreReturn.second,
              n,
              nullptr);
        } else { /* (2) Try normal token */
          auto lmScoreReturn = lm_->score(prevHyp.lmState, n);
          candidatesAdd(
              lmScoreReturn.first,
              &prevHyp,
              score + opt_.lmWeight * lmScoreReturn.second,
              n,
              outState);
        }
      }
      validHypo++;
    }
    candidatesStore(hyp_[t + 1], true);
    updateLMCache(lm_, hyp_[t + 1]);

  } // End of decoding

  while (t > 0 && hyp_[t].empty()) {
    --t;
  }
  hyp_[maxOutputLength_ + 1].resize(hyp_[t].size());
  for (int i = 0; i < hyp_[t].size(); i++) {
    hyp_[maxOutputLength_ + 1][i] = std::move(hyp_[t][i]);
  }
}

std::vector<DecodeResult> LexiconFreeSeq2SeqDecoder::getAllFinalHypothesis()
    const {
  return getAllHypothesis(hyp_.find(maxOutputLength_ + 1)->second, hyp_.size());
}

DecodeResult LexiconFreeSeq2SeqDecoder::getBestHypothesis(
    int /* unused */) const {
  return getHypothesis(
      hyp_.find(maxOutputLength_ + 1)->second.data(), hyp_.size());
}

void LexiconFreeSeq2SeqDecoder::prune(int /* unused */) {
  return;
}

int LexiconFreeSeq2SeqDecoder::nDecodedFramesInBuffer() const {
  /* unused function */
  return -1;
}

} // namespace w2l
