/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <iostream>

#include "common/Defines.h"
#include "common/Utils.h"
#include "lm/ConvLM.h"
#include "runtime/Serial.h"

namespace w2l {

// TODO: ConvLM should not depend on `runtime/Serial.h`. This ctor should take
// fl::Module and Dictionary directly instead of path strings.

ConvLM::ConvLM(
    const std::string& modelPath,
    const std::string& tokenVocabPath,
    const Dictionary& usrTknDict,
    int lmMemory,
    int beamSize,
    int historySize)
    : lmMemory_(lmMemory), beamSize_(beamSize), maxHistorySize_(historySize) {
  if (historySize < 1) {
    throw std::invalid_argument("[ConvLM] History size is too small.");
  }

  if (!fileExists(modelPath)) {
    throw std::runtime_error(
        "[ConvLM] File with ConvLM model '" + modelPath + "' doesn't exist");
  }

  /* Load token vocabulary */
  // Note: fairseq vocab should start with:
  // <fairseq_style> - 0 <pad> - 1, </s> - 2, <unk> - 3
  std::cerr << "[ConvLM]: Loading vocabulary from " << tokenVocabPath << "\n";
  vocab_ = Dictionary(tokenVocabPath);
  vocab_.setDefaultIndex(vocab_.getIndex(kUnkToken));
  vocabSize_ = vocab_.indexSize();
  std::cerr << "[ConvLM]: vocabulary size of convLM " << vocabSize_ << "\n";

  /* Load LM */
  std::cerr << "[ConvLM]: Loading LM from " << modelPath << "\n";
  W2lSerializer::load(modelPath, network_);
  network_->eval();
  std::cerr << "[ConvLM]: Finish loading LM from " << modelPath << "\n";

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
  auto inState = getRawState(state);
  int inStateLength = inState->length;
  std::shared_ptr<ConvLMState> outState;

  // Prepare output state
  if (inStateLength == maxHistorySize_) {
    outState = std::make_shared<ConvLMState>(maxHistorySize_);
    std::copy(
        inState->tokens.begin() + 1,
        inState->tokens.end(),
        outState->tokens.begin());
    outState->tokens[maxHistorySize_ - 1] = tokenIdx;
  } else {
    outState = std::make_shared<ConvLMState>(inStateLength + 1);
    std::copy(
        inState->tokens.begin(),
        inState->tokens.end(),
        outState->tokens.begin());
    outState->tokens[inStateLength] = tokenIdx;
  }

  // Prepare score
  float score = 0;
  if (tokenIdx < 0 || tokenIdx >= vocabSize_) {
    throw std::out_of_range(
        "[ConvLM] Invalid query word: " + std::to_string(tokenIdx));
  }

  if (cacheIndices_.find(inState) != cacheIndices_.end()) {
    // Cache hit
    auto cacheInd = cacheIndices_[inState];
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
    cacheIndices_[inState] = newIdx;

    std::vector<int> lastTokenPositions = {inState->length - 1};
    cache_[newIdx] = getLogProb(inState->tokens, lastTokenPositions)[0];
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
    auto state_ = getRawState(state);
    if (cacheIndices_.find(state_) != cacheIndices_.end()) {
      slot_[cacheIndices_[state_]] = state_;
    } else if (state_->length > longestHistory) {
      // prepare intest history only for those which should be predicted
      longestHistory = state_->length;
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
      auto state = getRawState(states[i]);
      if (cacheIndices_.find(state) != cacheIndices_.end()) {
        continue;
      }
      cacheIndices_[state] = cacheSize + nBatchStates;
      int start = nBatchStates * longestHistory;

      for (int j = 0; j < state->length; j++) {
        batchedTokens_[start + j] = state->tokens[j];
      }
      start += state->length;
      for (int j = 0; j < longestHistory - state->length; j++) {
        batchedTokens_[start + j] = vocab_.getIndex(kLmPadToken);
      }
      lastTokenPositions.push_back(state->length - 1);
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
    auto batchedProb = getLogProb(
        batchedTokens_, lastTokenPositions, longestHistory, nBatchStates);

    // Place probabilities in cache
    for (int i = 0; i < nBatchStates; i++, cacheSize++) {
      if (batchedProb[i].size() != vocabSize_) {
        throw std::logic_error(
            "[ConvLM] Batch probability size " +
            std::to_string(batchedProb[i].size()) +
            " mismatch with vocab size " + std::to_string(vocabSize_));
      }
      std::memcpy(
          cache_[cacheSize].data(),
          batchedProb[i].data(),
          vocabSize_ * sizeof(float));
    }
  }
}

std::vector<std::vector<float>> ConvLM::getLogProb(
    const std::vector<int>& inputs,
    const std::vector<int>& lastTokenPositions,
    int sampleSize,
    int batchSize) {
  sampleSize = sampleSize > 0 ? sampleSize : inputs.size();
  if (sampleSize * batchSize > inputs.size()) {
    throw std::invalid_argument(
        "[ConvLM] Incorrect sample size (" + std::to_string(sampleSize) +
        ") or batch size (" + std::to_string(batchSize) + ").");
  }
  af::array inputData(sampleSize, batchSize, inputs.data());
  fl::Variable output = network_->forward({fl::input(inputData)})[0];

  if (af::count<int>(af::isNaN(output.array())) != 0) {
    throw std::runtime_error("[ConvLM] Encountered NaNs in propagation");
  }
  std::vector<std::vector<float>> chosenFramePred(batchSize);
  auto preds = af::reorder(output.array(), 2, 1, 0); // (b t c)
  if (preds.dims(0) != batchSize) {
    throw std::logic_error(
        "[ConvLM]: incorrect predictions: batch should be " +
        std::to_string(batchSize) + " but it is " +
        std::to_string(preds.dims(0)));
  }
  for (int idx = 0; idx < batchSize; idx++) {
    if ((lastTokenPositions[idx] < 0) ||
        (lastTokenPositions[idx] >= preds.dims(1))) {
      throw std::logic_error(
          "[ConvLM]: trying the access to batch idx " + std::to_string(idx) +
          " and time idx " + std::to_string(lastTokenPositions[idx]) +
          " while the sizes are b: " + std::to_string(preds.dims(0)) +
          " t: " + std::to_string(preds.dims(1)));
    }
    chosenFramePred[idx] =
        afToVector<float>(preds.row(idx).col(lastTokenPositions[idx]));
  }
  return chosenFramePred;
}

int ConvLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto inState1 = getRawState(state1);
  auto inState2 = getRawState(state2);
  if (inState1 == inState2) {
    return 0;
  }
  if (inState1->length != inState2->length) {
    return inState1->length < inState2->length ? -1 : 1;
  }
  return std::memcmp(
      inState1->tokens.data(),
      inState2->tokens.data(),
      inState1->length * sizeof(int));
}

ConvLMState* ConvLM::getRawState(const LMStatePtr& state) {
  return static_cast<ConvLMState*>(state.get());
}

} // namespace w2l
