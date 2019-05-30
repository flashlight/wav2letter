/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "lm/KenLM.h"

namespace w2l {

KenLM::KenLM(const std::string& path) {
  model = lm::ngram::LoadVirtual(path.c_str());
  if (!model) {
    LOG(FATAL) << "[KenLM] LM loading failed.";
  }
  vocab = &model->BaseVocabulary();
  if (!vocab) {
    LOG(FATAL) << "[KenLM] LM vocabulary loading failed.";
  }
}

int KenLM::index(const std::string& token) {
  return vocab->Index(token.c_str());
}

LMStatePtr KenLM::start(bool isNull) {
  lm::ngram::State outState;
  if (isNull) {
    model->NullContextWrite(&outState);
  } else {
    model->BeginSentenceWrite(&outState);
  }
  return std::make_shared<KenLMState>(outState);
}

LMStatePtr KenLM::score(const LMStatePtr& inState, int tokenIdx, float& score) {
  auto inState_ = static_cast<KenLMState*>(inState.get());
  lm::ngram::State outState;
  score = model->BaseScore(&inState_->state_, tokenIdx, &outState);
  return std::make_shared<KenLMState>(outState);
}

LMStatePtr KenLM::finish(const LMStatePtr& inState, float& score) {
  /* DEBUG: could skip the end sentence </s> */
  auto inState_ = static_cast<KenLMState*>(inState.get());
  lm::ngram::State outState;
  score = model->BaseScore(&inState_->state_, vocab->EndSentence(), &outState);
  return std::make_shared<KenLMState>(outState);
}

int KenLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto state1_ = static_cast<KenLMState*>(state1.get());
  auto state2_ = static_cast<KenLMState*>(state2.get());
  return state1_->state_.Compare(state2_->state_);
}

} // namespace w2l
