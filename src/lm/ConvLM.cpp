/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "lm/ConvLM.h"
#include "common/Defines.h"
#include "common/Utils-base.h"
#include "runtime/Serial.h"

namespace w2l {

ConvLM::ConvLM(
    const std::string& modelPath,
    const std::string& tokenVocabPath,
    int lmMemory,
    int beamSize,
    int historySize)
    : lmMemory_(lmMemory), beamSize_(beamSize), maxHistorySize_(historySize) {
  if (historySize < 1) {
    LOG(FATAL) << "[ConvLM] History size is too small.";
  }

  if (!fileExists(modelPath)) {
    LOG(FATAL) << "[ConvLM] File with ConvLM model '" << modelPath
               << "' doesn't exist";
  }
  /* Load token vocabulary */
  // fairseq vocab should be <fairseq_style> - 0 <pad> - 1, </s> - 2, <unk> - 3

  LOG(INFO) << "[ConvLM]: Loading vocabulary from " << tokenVocabPath;
  vocab_ = createLMDict(tokenVocabPath);
  vocab_.setDefaultIndex(vocab_.getIndex(kUnkToken));
  vocabSize_ = vocab_.indexSize();
  LOG(INFO) << "[ConvLM]: vocabulary size of convLM " << vocabSize_;

  /* Load LM */
  LOG(INFO) << "[ConvLM]: Loading LM from " << modelPath;
  W2lSerializer::load(modelPath, network_);
  network_->eval();
  LOG(INFO) << "[ConvLM]: Finish loading LM from " << modelPath;

  /* Refresh cache */
  cacheIndices_.reserve(beamSize_);
  cache_.resize(beamSize_, std::vector<float>(vocabSize_));
  slot_.reserve(beamSize_);
  batchedTokens_.resize(beamSize_ * maxHistorySize_);
}

int ConvLM::index(const std::string& token) {
  return vocab_.getIndex(token.c_str());
}

LMStatePtr ConvLM::start(bool startWithNonEos) {
  auto outState = std::make_shared<ConvLMState>(1);
  if (!startWithNonEos) {
    outState->length = 1;
    outState->tokens[0] = index(kLmEosToken);
  } else {
    LOG(FATAL) << "[ConvLM] Only support using EOS to start the sentence";
  }
  return outState;
}

LMStatePtr
ConvLM::score(const LMStatePtr& inState, int tokenIdx, float& scoreRef) {
  const ConvLMState* inState_ = static_cast<ConvLMState*>(inState.get());
  int inStateLength = inState_->length;
  std::shared_ptr<ConvLMState> outState;

  // Prepare output state
  if (inStateLength == maxHistorySize_) {
    outState = std::make_shared<ConvLMState>(maxHistorySize_);
    std::copy(
        inState_->tokens.begin() + 1,
        inState_->tokens.end(),
        outState->tokens.begin());
    outState->tokens[maxHistorySize_ - 1] = tokenIdx;
  } else {
    outState = std::make_shared<ConvLMState>(inStateLength + 1);
    std::copy(
        inState_->tokens.begin(),
        inState_->tokens.end(),
        outState->tokens.begin());
    outState->tokens[inStateLength] = tokenIdx;
  }

  // Prepare score
  if (tokenIdx < 0 || tokenIdx >= vocabSize_) {
    LOG(FATAL) << "[ConvLM] Invalid query word: " << tokenIdx;
  }

  if (cacheIndices_.find(inState_) != cacheIndices_.end()) {
    // Cache hit
    auto cacheInd = cacheIndices_[inState_];
    if (cacheInd < 0 || cacheInd >= beamSize_) {
      LOG(FATAL) << "[ConvLM] Invalid cache access: " << cacheInd;
    }
    scoreRef = cache_[cacheInd][tokenIdx];
  } else {
    // Cache miss
    if (cacheIndices_.size() == beamSize_) {
      cacheIndices_.clear();
    }
    int newIdx = cacheIndices_.size();
    cacheIndices_[inState_] = newIdx;

    std::vector<int> lastTokenPositions = {inState_->length - 1};
    cache_[newIdx] = getLogProb(inState_->tokens, lastTokenPositions)[0];
    scoreRef = cache_[newIdx][tokenIdx];
  }
  if (std::isnan(scoreRef) || !std::isfinite(scoreRef)) {
    LOG(FATAL) << "[ConvLM] Wrong scoring from ConvLM: " << scoreRef;
  }
  return outState;
}

void ConvLM::updateCache(std::vector<LMStatePtr> states) {
  int longestHistory = -1, nStates = states.size();
  if (nStates > beamSize_) {
    LOG(FATAL)
        << "[ConvLM] Cache size too small (consider larger than beam size).";
  }

  // Refresh cache, store LM states that did not changed
  slot_.clear();
  slot_.resize(beamSize_, nullptr);
  for (const auto& state : states) {
    const ConvLMState* state_ = static_cast<ConvLMState*>(state.get());
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
      const ConvLMState* state = static_cast<ConvLMState*>(states[i].get());
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
        batchedTokens_[start + j] = index(kLmPadToken);
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
      LOG(FATAL) << "[ConvLM] Invalid batch: [" << nBatchStates << " x "
                 << longestHistory << "]";
    }
    auto batchedProb = getLogProb(
        batchedTokens_, lastTokenPositions, longestHistory, nBatchStates);

    // Place probabilities in cache
    for (int i = 0; i < nBatchStates; i++, cacheSize++) {
      if (batchedProb[i].size() != vocabSize_) {
        LOG(FATAL) << "[ConvLM] Batch probability size "
                   << batchedProb[i].size() << " mismatch with vocab size "
                   << vocabSize_;
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
    LOG(FATAL) << "[ConvLm] Incorrect sample size (" << sampleSize
               << ") / batch size (" << batchSize << ").";
  }
  af::array inputData(sampleSize, batchSize, inputs.data());
  fl::Variable output = network_->forward({fl::input(inputData)})[0];

  if (af::count<int>(af::isNaN(output.array())) != 0) {
    LOG(FATAL) << "Wrong propagation";
  };
  std::vector<std::vector<float>> chosenFramePred(batchSize);
  auto preds = af::reorder(output.array(), 2, 1, 0); // (b t c)
  if (preds.dims(0) != batchSize) {
    LOG(FATAL) << "[ConvLM]: incorrect predictions: batch should be "
               << batchSize << " but it is " << preds.dims(0);
  }
  for (int idx = 0; idx < batchSize; idx++) {
    if ((lastTokenPositions[idx] < 0) ||
        (lastTokenPositions[idx] >= preds.dims(1))) {
      LOG(FATAL) << "[ConvLM]: trying the access to batch idx " << idx
                 << " and time idx " << lastTokenPositions[idx]
                 << " while thwe sizes are b: " << preds.dims(0)
                 << " t: " << preds.dims(1);
    }
    chosenFramePred[idx] =
        afToVector<float>(preds.row(idx).col(lastTokenPositions[idx]));
  }
  return chosenFramePred;
}

LMStatePtr ConvLM::finish(const LMStatePtr& inState, float& scoreRef) {
  return score(inState, index(kLmEosToken), scoreRef);
}

int ConvLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto state1_ = static_cast<ConvLMState*>(state1.get());
  auto state2_ = static_cast<ConvLMState*>(state2.get());
  if (state1_->length != state2_->length) {
    return state1_->length < state2_->length ? -1 : 1;
  }
  return std::memcmp(
      state1_->tokens.data(),
      state2_->tokens.data(),
      state1_->length * sizeof(int));
}

} // namespace w2l
