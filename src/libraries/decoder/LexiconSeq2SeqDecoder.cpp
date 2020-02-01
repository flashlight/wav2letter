/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>

#include "libraries/decoder/LexiconSeq2SeqDecoder.h"
#include "libraries/lm/KenLM.h"
#include "libraries/lm/ZeroLM.h"

namespace w2l {

void LexiconSeq2SeqDecoder::decodeStep(const float* emissions, int T, int N) {
  // Extend hyp_ buffer
  if (hyp_.size() < maxOutputLength_ + 2) {
    for (int i = hyp_.size(); i < maxOutputLength_ + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconSeq2SeqDecoderState>());
    }
  }

  // Start from here.
  hyp_[0].clear();
  hyp_[0].emplace_back(
      0.0, lm_->start(0), lexicon_->getRoot(), nullptr, -1, -1, nullptr);

  auto compare = [](const LexiconSeq2SeqDecoderState& n1,
                    const LexiconSeq2SeqDecoderState& n2) {
    return n1.score > n2.score;
  };

  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset(candidatesBestScore_, candidates_, candidatePtrs_);

    // Batch forwarding
    rawY_.clear();
    rawPrevStates_.clear();
    for (const LexiconSeq2SeqDecoderState& prevHyp : hyp_[t]) {
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
      const LexiconSeq2SeqDecoderState& prevHyp = hyp_[t][hypo];
      // Change nothing for completed hypothesis
      if (prevHyp.token == eos_) {
        candidatesAdd(
            candidates_,
            candidatesBestScore_,
            opt_.beamThreshold,
            prevHyp.score,
            prevHyp.lmState,
            prevHyp.lex,
            &prevHyp,
            eos_,
            -1,
            nullptr,
            prevHyp.amScore,
            prevHyp.lmScore);
        continue;
      }

      const AMStatePtr& outState = outStates[validHypo];
      if (!outState) {
        validHypo++;
        continue;
      }

      const TrieNode* prevLex = prevHyp.lex;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot() ? 0 : prevLex->maxScore;

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

        /* (1) Try eos */
        if (n == eos_ && (prevLex == lexicon_->getRoot())) {
          auto lmStateScorePair = lm_->finish(prevHyp.lmState);
          LMStatePtr lmState = lmStateScorePair.first;
          double lmScore;
          if (isLmToken_) {
            lmScore = lmStateScorePair.second;
          } else {
            lmScore = lmStateScorePair.second - lexMaxScore;
          }

          candidatesAdd(
              candidates_,
              candidatesBestScore_,
              opt_.beamThreshold,
              prevHyp.score + amScore + opt_.eosScore + opt_.lmWeight * lmScore,
              lmState,
              lexicon_->getRoot(),
              &prevHyp,
              n,
              -1,
              nullptr,
              prevHyp.amScore + amScore,
              prevHyp.lmScore + lmScore);
        }

        /* (2) Try normal token */
        if (n != eos_) {
          auto searchLex = prevLex->children.find(n);
          if (searchLex != prevLex->children.end()) {
            auto lex = searchLex->second;
            LMStatePtr lmState;
            double lmScore;
            if (isLmToken_) {
              auto lmStateScorePair = lm_->score(prevHyp.lmState, n);
              lmState = lmStateScorePair.first;
              lmScore = lmStateScorePair.second;
            } else {
              // smearing
              lmState = prevHyp.lmState;
              lmScore = lex->maxScore - lexMaxScore;
            }
            candidatesAdd(
                candidates_,
                candidatesBestScore_,
                opt_.beamThreshold,
                prevHyp.score + amScore + opt_.lmWeight * lmScore,
                lmState,
                lex.get(),
                &prevHyp,
                n,
                -1,
                outState,
                prevHyp.amScore + amScore,
                prevHyp.lmScore + lmScore);

            // If we got a true word
            if (lex->labels.size() > 0) {
              for (auto word : lex->labels) {
                if (!isLmToken_) {
                  auto lmStateScorePair = lm_->score(prevHyp.lmState, word);
                  lmState = lmStateScorePair.first;
                  lmScore = lmStateScorePair.second - lexMaxScore;
                }
                candidatesAdd(
                    candidates_,
                    candidatesBestScore_,
                    opt_.beamThreshold,
                    prevHyp.score + amScore + opt_.wordScore +
                        opt_.lmWeight * lmScore,
                    lmState,
                    lexicon_->getRoot(),
                    &prevHyp,
                    n,
                    word,
                    outState,
                    prevHyp.amScore + amScore,
                    prevHyp.lmScore + lmScore);
                if (isLmToken_) {
                  break;
                }
              }
            }
          }
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
  } // End of decoding

  while (t > 0 && hyp_[t].empty()) {
    --t;
  }
  hyp_[maxOutputLength_ + 1].resize(hyp_[t].size());
  for (int i = 0; i < hyp_[t].size(); i++) {
    hyp_[maxOutputLength_ + 1][i] = std::move(hyp_[t][i]);
  }
}

std::vector<DecodeResult> LexiconSeq2SeqDecoder::getAllFinalHypothesis() const {
  return getAllHypothesis(hyp_.find(maxOutputLength_ + 1)->second, hyp_.size());
}

DecodeResult LexiconSeq2SeqDecoder::getBestHypothesis(int /* unused */) const {
  return getHypothesis(
      hyp_.find(maxOutputLength_ + 1)->second.data(), hyp_.size());
}

void LexiconSeq2SeqDecoder::prune(int /* unused */) {
  return;
}

int LexiconSeq2SeqDecoder::nDecodedFramesInBuffer() const {
  /* unused function */
  return -1;
}

} // namespace w2l
