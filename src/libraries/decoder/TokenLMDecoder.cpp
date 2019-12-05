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
#include <unordered_map>

#include "libraries/decoder/TokenLMDecoder.h"

namespace w2l {

void TokenLMDecoder::mergeCandidates() {
  auto compareNodesShortList = [&](const LexiconDecoderState* node1,
                                   const LexiconDecoderState* node2) {
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

void TokenLMDecoder::decodeStep(const float* emissions, int T, int N) {
  int startFrame = nDecodedFrames_ - nPrunedFrames_;
  // Extend hyp_ buffer
  if (hyp_.size() < startFrame + T + 2) {
    for (int i = hyp_.size(); i < startFrame + T + 2; i++) {
      hyp_.emplace(i, std::vector<LexiconDecoderState>());
    }
  }

  // Looping over all the frames
  for (int t = 0; t < T; t++) {
    candidatesReset();
    for (const LexiconDecoderState& prevHyp : hyp_[startFrame + t]) {
      const LMStatePtr& prevLmState = prevHyp.lmState;
      const TrieNode* prevLex = prevHyp.lex;
      const int prevIdx = prevLex->idx;

      /* (1) Try children */
      for (auto& child : prevLex->children) {
        int n = child.first;
        const TrieNode* lex = child.second.get();
        double score = prevHyp.score + emissions[t * N + n];
        if (nDecodedFrames_ + t > 0 &&
            opt_.criterionType == CriterionType::ASG) {
          score += transitions_[n * N + prevIdx];
        }
        if (n == sil_) {
          score += opt_.silWeight;
        }

        auto lmScoreReturn = lm_->score(prevLmState, n);
        score += lmScoreReturn.second * opt_.lmWeight;

        // We eat-up a new token
        if (opt_.criterionType != CriterionType::CTC || prevHyp.prevBlank ||
            n != prevIdx) {
          if (!lex->children.empty()) {
            candidatesAdd(
                lmScoreReturn.first,
                lex,
                &prevHyp,
                score,
                n,
                -1,
                false // prevBlank
            );
          }
        }

        // If we got a true word
        for (auto label : lex->labels) {
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.wordScore,
              n,
              label,
              false // prevBlank
          );
        }

        // If we got an unknown word and we want to emit
        if (lex->labels.empty() && (opt_.unkScore > kNegativeInfinity)) {
          candidatesAdd(
              lmScoreReturn.first,
              lexicon_->getRoot(),
              &prevHyp,
              score + opt_.unkScore,
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
          score += opt_.silWeight;
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
    }

    candidatesStore(hyp_[startFrame + t + 1], false);
    updateLMCache(lm_, hyp_[startFrame + t + 1]);
  }
  nDecodedFrames_ += T;
}

} // namespace w2l
