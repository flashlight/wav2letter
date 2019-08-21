/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/lm/ZeroLM.h"

#include <stdexcept>

namespace w2l {

LMStatePtr ZeroLM::start(bool /* unused */) {
  return std::make_shared<ZeroLMState>(-1);
}

std::pair<LMStatePtr, float> ZeroLM::score(
    const LMStatePtr& /* unused */,
    const int usrTokenIdx) {
  return std::make_pair(std::make_shared<ZeroLMState>(usrTokenIdx), 0.0);
}

std::pair<LMStatePtr, float> ZeroLM::finish(const LMStatePtr& /* unused */) {
  return std::make_pair(std::make_shared<ZeroLMState>(-1), 0.0);
}

int ZeroLM::compareState(const LMStatePtr& state1, const LMStatePtr& state2)
    const {
  auto inState1 = getRawState(state1);
  auto inState2 = getRawState(state2);
  if (inState1->token == inState2->token) {
    return 0;
  }
  return inState1->token < inState2->token ? -1 : 1;
}

ZeroLMState* ZeroLM::getRawState(const LMStatePtr& state) {
  return static_cast<ZeroLMState*>(state.get());
}

} // namespace w2l
