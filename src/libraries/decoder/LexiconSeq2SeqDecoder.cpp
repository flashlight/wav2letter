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
#include <numeric>

#include "libraries/decoder/LexiconSeq2SeqDecoder.h"
#include "libraries/lm/KenLM.h"
#include "libraries/lm/ZeroLM.h"

namespace w2l {

void LexiconSeq2SeqDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_.clear();
  candidatePtrs_.clear();
}

void LexiconSeq2SeqDecoder::mergeCandidates() {
  auto compareNodesShortList = [](const LexiconSeq2SeqDecoderState* node1,
                                  const LexiconSeq2SeqDecoderState* node2) {
    int lmCmp = node1->lmState->compare(node2->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else if (node1->lex != node2->lex) {
      return node1->lex > node2->lex;
    } else if (node1->token != node2->token) {
      return node1->token > node2->token;
    } else {
      /* same LmState, same lex */
      return node1->score > node2->score;
    }
  };
  std::sort(
      candidatePtrs_.begin(), candidatePtrs_.end(), compareNodesShortList);

  int nHypAfterMerging = 1;
  for (int i = 1; i < candidatePtrs_.size(); i++) {
    if (candidatePtrs_[i]->lmState->compare(
            candidatePtrs_[nHypAfterMerging - 1]->lmState) ||
        candidatePtrs_[i]->lex != candidatePtrs_[nHypAfterMerging - 1]->lex ||
        candidatePtrs_[i]->word != candidatePtrs_[nHypAfterMerging - 1]->word ||
        candidatePtrs_[i]->token !=
            candidatePtrs_[nHypAfterMerging - 1]->token) {
      candidatePtrs_[nHypAfterMerging] = candidatePtrs_[i];
      nHypAfterMerging++;
    } else {
      mergeStates(
          candidatePtrs_[nHypAfterMerging - 1], candidatePtrs_[i], opt_.logAdd);
    }
  }

  candidatePtrs_.resize(nHypAfterMerging);
}

void LexiconSeq2SeqDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const TrieNode* lex,
    const LexiconSeq2SeqDecoderState* parent,
    const double score,
    const int token,
    const int word,
    const AMStatePtr& amState) {
  if (isValidCandidate(candidatesBestScore_, score, opt_.beamThreshold)) {
    candidates_.emplace_back(LexiconSeq2SeqDecoderState(
        lmState, lex, parent, score, token, word, amState));
  }
}

void LexiconSeq2SeqDecoder::candidatesStore(
    std::vector<LexiconSeq2SeqDecoderState>& nextHyp,
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
      lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, -1, -1, nullptr);

  auto compare = [](const LexiconSeq2SeqDecoderState& n1,
                    const LexiconSeq2SeqDecoderState& n2) {
    return n1.score > n2.score;
  };

  // Decode frame by frame
  int t = 0;
  for (; t < maxOutputLength_; t++) {
    candidatesReset();

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
            prevHyp.lmState,
            prevHyp.lex,
            &prevHyp,
            prevHyp.score,
            eos_,
            -1,
            nullptr);
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
        double score = prevHyp.score + amScores[validHypo][n];

        /* (1) Try eos */
        if (n == eos_ && (prevLex == lexicon_->getRoot())) {
          auto lmReturn = lm_->finish(prevHyp.lmState);
          LMStatePtr lmState = lmReturn.first;
          double lmScore;
          if (isLmToken_) {
            lmScore = lmReturn.second;
          } else {
            lmScore = lmReturn.second - lexMaxScore;
          }
          candidatesAdd(
              lmState,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.eosScore + opt_.lmWeight * lmScore,
              n,
              -1,
              nullptr);
        }

        /* (2) Try normal token */
        if (n != eos_) {
          auto searchLex = prevLex->children.find(n);
          if (searchLex != prevLex->children.end()) {
            auto lex = searchLex->second;
            LMStatePtr lmState;
            double lmScore;
            if (isLmToken_) {
              auto lmReturn = lm_->score(prevHyp.lmState, n);
              lmState = lmReturn.first;
              lmScore = lmReturn.second;
            } else {
              // smearing
              lmState = prevHyp.lmState;
              lmScore = lex->maxScore - lexMaxScore;
            }
            candidatesAdd(
                lmState,
                lex.get(),
                &prevHyp,
                score + opt_.lmWeight * lmScore,
                n,
                -1,
                outState);
            if (lex->labels.size() > 0) {
              for (auto word : lex->labels) {
                if (!isLmToken_) {
                  auto lmReturn = lm_->score(prevHyp.lmState, word);
                  lmState = lmReturn.first;
                  lmScore = lmReturn.second - lexMaxScore;
                }
                candidatesAdd(
                    lmState,
                    lexicon_->getRoot(),
                    &prevHyp,
                    score + opt_.wordScore + opt_.lmWeight * lmScore,
                    n,
                    word,
                    outState);
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
