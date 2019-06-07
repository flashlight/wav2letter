/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "common/Dictionary.h"
#include "decoder/LM.h"
#include "lm/model.hh"

namespace w2l {
/**
 * KenLMState is a state object from KenLM, which  contains context length,
 * indicies and compare functions
 * https://github.com/kpu/kenlm/blob/master/lm/state.hh.
 */
using KenLMState = lm::ngram::State;

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class KenLM : public LM {
 public:
  KenLM(const std::string& path, const Dictionary& usrTknDict);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override;

 private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary* vocab_;

  static KenLMState* getRawState(const LMStatePtr& state);
};

} // namespace w2l
