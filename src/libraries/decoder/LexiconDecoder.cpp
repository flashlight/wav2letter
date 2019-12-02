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

#include "libraries/decoder/LexiconDecoder.h"

namespace w2l {

void LexiconDecoder::candidatesReset() {
  candidatesBestScore_ = kNegativeInfinity;
  candidates_.clear();
  candidatePtrs_.clear();
}

void LexiconDecoder::candidatesAdd(
    const LMStatePtr& lmState,
    const TrieNode* lex,
    const LexiconDecoderState* parent,
    const double score,
    const int token,
    const int word,
    const bool prevBlank) {
  if (isValidCandidate(candidatesBestScore_, score, opt_.beamThreshold)) {
    candidates_.emplace_back(
        lmState, lex, parent, score, token, word, prevBlank);
  }
}

void LexiconDecoder::candidatesStore(
    std::vector<LexiconDecoderState>& nextHyp,
    const bool returnSorted) {
  if (candidates_.empty()) {
    nextHyp.clear();
    return;
  }

  /* Select valid candidates */
  pruneCandidates(
      candidatePtrs_, candidates_, candidatesBestScore_ - opt_.beamThreshold);

  /* Sort by (lmState, lex, score) and copy into next hypothesis */
  mergeCandidates();

  /* Sort hypothesis and select top-K */
  storeTopCandidates(nextHyp, candidatePtrs_, opt_.beamSize, returnSorted);
}

void LexiconDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<LexiconDecoderState>());

  /* note: the lm reset itself with :start() */
  hyp_[0].emplace_back(
      lm_->start(0), lexicon_->getRoot(), nullptr, 0.0, sil_, -1);
  nDecodedFrames_ = 0;
  nPrunedFrames_ = 0;
}

void LexiconDecoder::decodeEnd() {
  candidatesReset();
  for (const LexiconDecoderState& prevHyp :
       hyp_[nDecodedFrames_ - nPrunedFrames_]) {
    const TrieNode* prevLex = prevHyp.lex;
    const LMStatePtr& prevLmState = prevHyp.lmState;

    auto lmStateScorePair = lm_->finish(prevLmState);
    candidatesAdd(
        lmStateScorePair.first,
        prevLex,
        &prevHyp,
        prevHyp.score + opt_.lmWeight * lmStateScorePair.second,
        sil_,
        -1,
        false // prevBlank
    );
  }

  candidatesStore(hyp_[nDecodedFrames_ - nPrunedFrames_ + 1], true);
  ++nDecodedFrames_;
}

std::vector<DecodeResult> LexiconDecoder::getAllFinalHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  if (finalFrame < 1) {
    return std::vector<DecodeResult>{};
  }

  return getAllHypothesis(hyp_.find(finalFrame)->second, finalFrame);
}

DecodeResult LexiconDecoder::getBestHypothesis(int lookBack) const {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return DecodeResult();
  }

  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
  return getHypothesis(bestNode, nDecodedFrames_ - nPrunedFrames_ - lookBack);
}

int LexiconDecoder::nHypothesis() const {
  int finalFrame = nDecodedFrames_ - nPrunedFrames_;
  return hyp_.find(finalFrame)->second.size();
}

int LexiconDecoder::nDecodedFramesInBuffer() const {
  return nDecodedFrames_ - nPrunedFrames_ + 1;
}

void LexiconDecoder::prune(int lookBack) {
  if (nDecodedFrames_ - nPrunedFrames_ - lookBack < 1) {
    return; // Not enough decoded frames to prune
  }

  /* (1) Find the last emitted word in the best path */
  const LexiconDecoderState* bestNode = findBestAncestor(
      hyp_.find(nDecodedFrames_ - nPrunedFrames_)->second, lookBack);
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
