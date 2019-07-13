/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/lm/KenLM.h"

#include <stdexcept>

#include <lm/model.hh>

namespace w2l {

KenLM::KenLM(const std::string& path, const Dictionary& usrTknDict) {
  // Load LM
  model_.reset(lm::ngram::LoadVirtual(path.c_str()));
  if (!model_) {
    throw std::runtime_error("[KenLM] LM loading failed.");
  }
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    throw std::runtime_error("[KenLM] LM vocabulary loading failed.");
  }

  // Create index map
  usrToLmIdxMap_.resize(usrTknDict.indexSize());
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getEntry(i);
    int lmIdx = vocab_->Index(token.c_str());
    usrToLmIdxMap_[i] = lmIdx;
  }
}

LMStatePtr KenLM::start(bool startWithNothing) {
  auto outState = std::make_shared<KenLMState>();
  if (startWithNothing) {
    model_->NullContextWrite(outState.get());
  } else {
    model_->BeginSentenceWrite(outState.get());
  }

  return outState;
}

std::pair<LMStatePtr, float> KenLM::score(
    const LMStatePtr& state,
    const int usrTokenIdx) {
  if (usrTokenIdx < 0 || usrTokenIdx >= usrToLmIdxMap_.size()) {
    throw std::runtime_error(
        "[KenLM] Invalid user token index: " + std::to_string(usrTokenIdx));
  }
  auto inState = getRawState(state);
  auto outState = std::make_shared<KenLMState>();
  float score =
      model_->BaseScore(inState, usrToLmIdxMap_[usrTokenIdx], outState.get());
  return std::make_pair(std::move(outState), score);
}

std::pair<LMStatePtr, float> KenLM::finish(const LMStatePtr& state) {
  auto inState = getRawState(state);
  auto outState = std::make_shared<KenLMState>();
  float score =
      model_->BaseScore(inState, vocab_->EndSentence(), outState.get());
  return std::make_pair(std::move(outState), score);
}

int KenLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto inState1 = getRawState(state1);
  auto inState2 = getRawState(state2);
  if (inState1 == inState2) {
    return 0;
  }
  return inState1->Compare(*inState2);
}

KenLMState* KenLM::getRawState(const LMStatePtr& state) {
  return static_cast<KenLMState*>(state.get());
}

} // namespace w2l
