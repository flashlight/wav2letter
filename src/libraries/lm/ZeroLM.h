/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libraries/lm/LM.h"

namespace w2l {

struct ZeroLMState {
  int token;

  explicit ZeroLMState(int id) : token(id) {}
};

/**
 * ZeroLM is a dummy language model class, which mimics the behavious of a
 * uni-gram language model but always returns 0 as score.
 */
class ZeroLM : public LM {
 public:
  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override;

 private:
  static ZeroLMState* getRawState(const LMStatePtr& state);
};

using ZeroLMPtr = std::shared_ptr<ZeroLM>;

} // namespace w2l
