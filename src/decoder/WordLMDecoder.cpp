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

#include "decoder/WordLMDecoder.h"

namespace w2l {

int WordLMDecoder::mergeCandidates(const int size) {
  auto compareNodesShortList = [&](const LexiconDecoderState* node1,
                                   const LexiconDecoderState* node2) {
    int lmCmp = lm_->compareState(node1->lmState_, node2->lmState_);
    if (lmCmp != 0) {
      return lmCmp > 0;
    } else if (node1->lex_ != node2->lex_) {
      /* same LmState */
      return node1->lex_ > node2->lex_;
    } else {
      /* same LmState, same lex */
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

void WordLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
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
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType_ == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight_;
        }

        // We eat-up a new token
        if (opt_.criterionType_ != CriterionType::CTC || prevHyp.prevBlank_ ||
            n != prevIdx) {
          if (!lex->children_.empty()) {
            candidatesAdd(
                prevLmState,
                lex.get(),
                &prevHyp,
                score + opt_.lmWeight_ * (lex->maxScore_ - lexMaxScore),
                n,
                -1,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (int i = 0; i < lex->nLabel_; i++) {
          float lmScore;
          LMStatePtr newLmState;
          std::tie(newLmState, lmScore) =
              lm_->score(prevLmState, lex->label_[i]);
          candidatesAdd(
              newLmState,
              lexicon_->getRoot().get(),
              &prevHyp,
              score + opt_.lmWeight_ * (lmScore - lexMaxScore) +
                  opt_.wordScore_,
              n,
              lex->label_[i],
              false // prevBlank
          );
        }

        // If we got an unknown word
        if (lex->nLabel_ == 0 && (opt_.unkScore_ > kNegativeInfinity)) {
          float lmScore;
          LMStatePtr newLmState;
          std::tie(newLmState, lmScore) = lm_->score(prevLmState, unk_);
          candidatesAdd(
              newLmState,
              lexicon_->getRoot().get(),
              &prevHyp,
              score + opt_.lmWeight_ * (lmScore - lexMaxScore) + opt_.unkScore_,
              n,
              unk_,
              false // prevBlank
          );
        }
      }

      /* (2) Try same lexicon node */
      if (opt_.criterionType_ != CriterionType::CTC || !prevHyp.prevBlank_) {
        int n = prevIdx;
        float score = prevHyp.score_ + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType_ == CriterionType::ASG) {
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
            -1,
            false // prevBlank
        );
      }

      /* (3) CTC only, try blank */
      if (opt_.criterionType_ == CriterionType::CTC) {
        int n = blank_;
        float score = prevHyp.score_ + emissions[t * N + n];
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
