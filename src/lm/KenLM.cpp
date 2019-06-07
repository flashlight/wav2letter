/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>

#include "lm/KenLM.h"

namespace w2l {

KenLM::KenLM(const std::string& path, const Dictionary& usrTknDict) {
  // Load LM
  model_.reset(lm::ngram::LoadVirtual(path.c_str()));
  if (!model_) {
    LOG(FATAL) << "[KenLM] LM loading failed.";
  }
  vocab_ = &model_->BaseVocabulary();
  if (!vocab_) {
    LOG(FATAL) << "[KenLM] LM vocabulary loading failed.";
  }

  // Create index map
  usrToLmIdxMap_.clear();
  for (int i = 0; i < usrTknDict.indexSize(); i++) {
    auto token = usrTknDict.getToken(i);
    int lmIdx = vocab_->Index(token.c_str());
    usrToLmIdxMap_.emplace(i, lmIdx);
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
  if (usrToLmIdxMap_.find(usrTokenIdx) == usrToLmIdxMap_.end()) {
    LOG(FATAL) << "[KenLM] Invalid user token index" << usrTokenIdx;
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
  return inState1->Compare(*inState2);
}

KenLMState* KenLM::getRawState(const LMStatePtr& state) {
  return static_cast<KenLMState*>(state.get());
}

} // namespace w2l
