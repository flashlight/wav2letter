/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "libraries/lm/ConvLM.h"

#include <cmath>
#include <cstring>
#include <iostream>

namespace w2l {

ConvLM::ConvLM(
    const GetConvLmScoreFunc& getConvLmScoreFunc,
    const std::string& tokenVocabPath,
    const Dictionary& usrTknDict,
    int lmMemory,
    int beamSize,
    int historySize)
    : lmMemory_(lmMemory),
      beamSize_(beamSize),
      getConvLmScoreFunc_(getConvLmScoreFunc),
      maxHistorySize_(historySize) {
  if (historySize < 1) {
    throw std::invalid_argument("[ConvLM] History size is too small.");
  }

  /* Load token vocabulary */
  // Note: fairseq vocab should start with:
  // <fairseq_style> - 0 <pad> - 1, </s> - 2, <unk> - 3
  std::cerr << "[ConvLM]: Loading vocabulary from " << tokenVocabPath << "\n";
  vocab_ = Dictionary(tokenVocabPath);
  vocab_.setDefaultIndex(vocab_.getIndex(kUnkToken));
  vocabSize_ = vocab_.indexSize();
  std::cerr << "[ConvLM]: vocabulary size of convLM " << vocabSize_ << "\n";

  /* Create index map */
  usrToLmIdxMap_.resize(usrTknDict.indexSize());
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getEntry(i);
    int lmIdx = vocab_.getIndex(token.c_str());
    usrToLmIdxMap_[i] = lmIdx;
  }

  /* Refresh cache */
  cacheIndices_.reserve(beamSize_);
  cache_.resize(beamSize_, std::vector<float>(vocabSize_));
  slot_.reserve(beamSize_);
  batchedTokens_.resize(beamSize_ * maxHistorySize_);
}

LMStatePtr ConvLM::start(bool startWithNothing) {
  cacheIndices_.clear();
  auto outState = std::make_shared<ConvLMState>(1);
  if (!startWithNothing) {
    outState->length = 1;
    outState->tokens[0] = vocab_.getIndex(kLmEosToken);
  } else {
    throw std::invalid_argument(
        "[ConvLM] Only support using EOS to start the sentence");
  }
  return outState;
}

std::pair<LMStatePtr, float> ConvLM::scoreWithLmIdx(
    const LMStatePtr& state,
    const int tokenIdx) {
  auto rawInState = std::static_pointer_cast<ConvLMState>(state).get();
  int inStateLength = rawInState->length;
  std::shared_ptr<ConvLMState> outState;

  // Prepare output state
  if (inStateLength == maxHistorySize_) {
    outState = std::make_shared<ConvLMState>(maxHistorySize_);
    std::copy(
        rawInState->tokens.begin() + 1,
        rawInState->tokens.end(),
        outState->tokens.begin());
    outState->tokens[maxHistorySize_ - 1] = tokenIdx;
  } else {
    outState = std::make_shared<ConvLMState>(inStateLength + 1);
    std::copy(
        rawInState->tokens.begin(),
        rawInState->tokens.end(),
        outState->tokens.begin());
    outState->tokens[inStateLength] = tokenIdx;
  }

  // Prepare score
  float score = 0;
  if (tokenIdx < 0 || tokenIdx >= vocabSize_) {
    throw std::out_of_range(
        "[ConvLM] Invalid query word: " + std::to_string(tokenIdx));
  }

  if (cacheIndices_.find(rawInState) != cacheIndices_.end()) {
    // Cache hit
    auto cacheInd = cacheIndices_[rawInState];
    if (cacheInd < 0 || cacheInd >= beamSize_) {
      throw std::logic_error(
          "[ConvLM] Invalid cache access: " + std::to_string(cacheInd));
    }
    score = cache_[cacheInd][tokenIdx];
  } else {
    // Cache miss
    if (cacheIndices_.size() == beamSize_) {
      cacheIndices_.clear();
    }
    int newIdx = cacheIndices_.size();
    cacheIndices_[rawInState] = newIdx;

    std::vector<int> lastTokenPositions = {rawInState->length - 1};
    cache_[newIdx] =
        getConvLmScoreFunc_(rawInState->tokens, lastTokenPositions, -1, 1);
    score = cache_[newIdx][tokenIdx];
  }
  if (std::isnan(score) || !std::isfinite(score)) {
    throw std::runtime_error(
        "[ConvLM] Bad scoring from ConvLM: " + std::to_string(score));
  }
  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> ConvLM::score(
    const LMStatePtr& state,
    const int usrTokenIdx) {
  if (usrTokenIdx < 0 || usrTokenIdx >= usrToLmIdxMap_.size()) {
    throw std::out_of_range(
        "[KenLM] Invalid user token index: " + std::to_string(usrTokenIdx));
  }
  return scoreWithLmIdx(state, usrToLmIdxMap_[usrTokenIdx]);
}

std::pair<LMStatePtr, float> ConvLM::finish(const LMStatePtr& state) {
  return scoreWithLmIdx(state, vocab_.getIndex(kLmEosToken));
}

void ConvLM::updateCache(std::vector<LMStatePtr> states) {
  int longestHistory = -1, nStates = states.size();
  if (nStates > beamSize_) {
    throw std::invalid_argument(
        "[ConvLM] Cache size too small (consider larger than beam size).");
  }

  // Refresh cache, store LM states that did not changed
  slot_.clear();
  slot_.resize(beamSize_, nullptr);
  for (const auto& state : states) {
    auto rawState = std::static_pointer_cast<ConvLMState>(state).get();
    if (cacheIndices_.find(rawState) != cacheIndices_.end()) {
      slot_[cacheIndices_[rawState]] = rawState;
    } else if (rawState->length > longestHistory) {
      // prepare intest history only for those which should be predicted
      longestHistory = rawState->length;
    }
  }
  cacheIndices_.clear();
  int cacheSize = 0;
  for (int i = 0; i < beamSize_; i++) {
    if (!slot_[i]) {
      continue;
    }
    cache_[cacheSize] = cache_[i];
    cacheIndices_[slot_[i]] = cacheSize;
    ++cacheSize;
  }

  // Determine batchsize
  if (longestHistory <= 0) {
    return;
  }
  // batchSize * longestHistory = cacheSize;
  int maxBatchSize = lmMemory_ / longestHistory;
  if (maxBatchSize > nStates) {
    maxBatchSize = nStates;
  }

  // Run batch forward
  int batchStart = 0;
  while (batchStart < nStates) {
    // Select batch
    int nBatchStates = 0;
    std::vector<int> lastTokenPositions;
    for (int i = batchStart; (nBatchStates < maxBatchSize) && (i < nStates);
         i++, batchStart++) {
      auto rawState = std::static_pointer_cast<ConvLMState>(states[i]).get();
      if (cacheIndices_.find(rawState) != cacheIndices_.end()) {
        continue;
      }
      cacheIndices_[rawState] = cacheSize + nBatchStates;
      int start = nBatchStates * longestHistory;

      for (int j = 0; j < rawState->length; j++) {
        batchedTokens_[start + j] = rawState->tokens[j];
      }
      start += rawState->length;
      for (int j = 0; j < longestHistory - rawState->length; j++) {
        batchedTokens_[start + j] = vocab_.getIndex(kLmPadToken);
      }
      lastTokenPositions.push_back(rawState->length - 1);
      ++nBatchStates;
    }
    if (nBatchStates == 0 && batchStart >= nStates) {
      // if all states were skipped
      break;
    }

    // Feed forward
    if (nBatchStates < 1 || longestHistory < 1) {
      throw std::logic_error(
          "[ConvLM] Invalid batch: [" + std::to_string(nBatchStates) + " x " +
          std::to_string(longestHistory) + "]");
    }
    auto batchedProb = getConvLmScoreFunc_(
        batchedTokens_, lastTokenPositions, longestHistory, nBatchStates);

    if (batchedProb.size() != vocabSize_ * nBatchStates) {
      throw std::logic_error(
          "[ConvLM] Batch X Vocab size " + std::to_string(batchedProb.size()) +
          " mismatch with " + std::to_string(vocabSize_ * nBatchStates));
    }
    // Place probabilities in cache
    for (int i = 0; i < nBatchStates; i++, cacheSize++) {
      std::memcpy(
          cache_[cacheSize].data(),
          batchedProb.data() + vocabSize_ * i,
          vocabSize_ * sizeof(float));
    }
  }
}

} // namespace w2l
