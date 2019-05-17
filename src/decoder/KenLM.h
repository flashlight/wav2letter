/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <glog/logging.h>
#include <vector>
#include "LM.h"
#include "lm/model.hh"

namespace w2l {
/**
 * KenLMState extends LMState by adding a state object from KenLM, which
 * contains context length, indicies and compare functions
 * https://github.com/kpu/kenlm/blob/master/lm/state.hh.
 */
struct KenLMState : public LMState {
  lm::ngram::State state_;

  explicit KenLMState(lm::ngram::State state) : state_(state) {}
};

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class KenLM : public LM {
 public:
  int index(const std::string& token) override;

  LMStatePtr start(bool isNull) override;

  LMStatePtr score(const LMStatePtr& inState, int tokenIdx, float& score)
      override;

  LMStatePtr finish(const LMStatePtr& inState, float& score) override;

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override;

  explicit KenLM(const std::string& path);

 private:
  const lm::base::Model* model;
  const lm::base::Vocabulary* vocab;
};

typedef std::shared_ptr<KenLM> KenLMPtr;

} // namespace w2l
