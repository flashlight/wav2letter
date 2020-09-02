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
  hyp_[0].emplace_back(0.0, lm_->start(0), nullptr, -1, nullptr);
  completedCandidates_.clear();

  auto hypComparator = [](const LexiconFreeSeq2SeqDecoderState& state1,
                          const LexiconFreeSeq2SeqDecoderState& state2) {
    return state1.score > state2.score;
  };
  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);

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
      // all previous hypothesis are completed, add them to the
      // completedCandidates_ before exit the loop
      for (const LexiconFreeSeq2SeqDecoderState& prevHyp : hyp_[t]) {
        completedCandidates_.push_back(prevHyp);
      }
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
        // add to pool of completed hyps to avoid thresholding them in the
        // future (only for full beam)
        completedCandidates_.push_back(prevHyp);
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
        double amScore = amScores[validHypo][n];

        if (n == eos_) { /* (1) Try eos */
          auto lmStateScorePair = lm_->finish(prevHyp.lmState);
          auto lmScore = lmStateScorePair.second;

          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevHyp.score + amScore + opt_.eosScore + opt_.lmWeight * lmScore,
              lmStateScorePair.first,
              &prevHyp,
              n,
              nullptr,
              prevHyp.amScore + amScore,
              prevHyp.lmScore + lmScore);
        } else { /* (2) Try normal token */
          auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
          auto lmScore = lmStateScorePair.second;
          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevHyp.score + amScore + opt_.lmWeight * lmScore,
              lmStateScorePair.first,
              &prevHyp,
              n,
              outState,
              prevHyp.amScore + amScore,
              prevHyp.lmScore + lmScore);
        }
      }
      validHypo++;
    }
    candidatesStore(
        candidates_,
        candidatePtrs_,
        hyp_[t + 1],
        opt_.beamSize,
        candidatesBestScore_ - opt_.beamThreshold,
        opt_.logAdd,
        true);
    updateLMCache(lm_, hyp_[t + 1]);

    if (completedCandidates_.size() >= opt_.beamSize) {
      std::partial_sort(
          completedCandidates_.begin(),
          completedCandidates_.begin() + opt_.beamSize,
          completedCandidates_.end(),
          hypComparator);
      completedCandidates_.resize(opt_.beamSize);
    }
  } // End of decoding

  std::vector<LexiconFreeSeq2SeqDecoderState> finalCandidates;
  if (completedCandidates_.size() > 0) {
    std::partial_sort(
        completedCandidates_.begin(),
        completedCandidates_.begin() + opt_.beamSize,
        completedCandidates_.end(),
        hypComparator);
    completedCandidates_.resize(opt_.beamSize);
    finalCandidates = completedCandidates_;
  } else {
    while (t > 0 && hyp_[t].empty()) {
      --t;
    }
    finalCandidates = hyp_[t];
  }
  hyp_[maxOutputLength_ + 1].resize(finalCandidates.size());
  for (int i = 0; i < finalCandidates.size(); i++) {
    hyp_[maxOutputLength_ + 1][i] = std::move(finalCandidates[i]);
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
