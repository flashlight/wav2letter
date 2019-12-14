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
#include <unordered_map>

#include "libraries/decoder/WordLMDecoder.h"

namespace w2l {

void WordLMDecoder::mergeCandidates() {
  auto compareNodesShortList = [](const LexiconDecoderState* node1,
                                  const LexiconDecoderState* node2) {
    int lmCmp = node1->lmState->compare(node2->lmState);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else if (node1->lex != node2->lex) {
      /* same LmState */
      return node1->lex > node2->lex;
    } else if (node1->token != node2->token) {
      return node1->token > node2->token;
    } else if (node1->prevBlank != node2->prevBlank) {
      return node1->prevBlank > node2->prevBlank;
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

void WordLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  std::vector<size_t> idx(N);
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
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const TrieNode* prevLex = prevHyp.lex;
      const int prevIdx = prevHyp.token;
      const float lexMaxScore =
          prevLex == lexicon_->getRoot() ? 0 : prevLex->maxScore;
      const LMStatePtr& prevLmState = prevHyp.lmState;

      /* (1) Try children */
      for (int r = 0; r < std::min(opt_.beamSizeToken, N); ++r) {
        int n = idx[r];
        auto iter = prevLex->children.find(n);
        if (iter == prevLex->children.end()) {
          continue;
        }
        const TrieNodePtr& lex = iter->second;
        double score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silScore;
        }

        // We eat-up a new token
        if (opt_.criterionType != CriterionType::CTC || prevHyp.prevBlank ||
            n != prevIdx) {
          if (!lex->children.empty()) {
            candidatesAdd(
                prevLmState,
                lex.get(),
                &prevHyp,
                score + opt_.lmWeight * (lex->maxScore - lexMaxScore),
                n,
                -1,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (auto label : lex->labels) {
          auto lmScoreReturn = lm_->score(prevLmState, label);
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.lmWeight * (lmScoreReturn.second - lexMaxScore) +
                  opt_.wordScore,
              n,
              label,
              false // prevBlank
          );
        }

        // If we got an unknown word
        if (lex->labels.empty() && (opt_.unkScore > kNegativeInfinity)) {
          auto lmScoreReturn = lm_->score(prevLmState, unk_);
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.lmWeight * (lmScoreReturn.second - lexMaxScore) +
                  opt_.unkScore,
              n,
              unk_,
              false // prevBlank
          );
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.criterionType != CriterionType::CTC || !prevHyp.prevBlank) {
        int n = prevIdx;
        double score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silScore;
        }

        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            false // prevBlank
        );
      }

      /* (3) CTC only, try blank */
      if (opt_.criterionType == CriterionType::CTC) {
        int n = blank_;
        double score = prevHyp.score + emissions[t * N + n];
        candidatesAdd(
            prevLmState,
            prevLex,
            &prevHyp,
            score,
            n,
            -1,
            true // prevBlank
        );
      }
      // finish proposing
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }

  nDecodedFrames_ += T;
}

} // namespace w2l
