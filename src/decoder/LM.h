/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>

namespace w2l {
/**
 * LMState is a base class for language model states.
 */
struct LMState {
  LMState() = default;
  virtual ~LMState() = default;
};

typedef std::shared_ptr<LMState> LMStatePtr;

/**
 * LM is a thin wrapper for laguage models. We abstrct several common methods
 * here which can be shared for KenLM, ConvLM, RNNLM, etc.
 */
class LM {
 public:
  /* Return the index in the language model of a given token. */
  virtual int index(const std::string& token) = 0;

  /* Initialize or reset language model */
  virtual LMStatePtr start(bool isNull) = 0;

  /**
   * Query the language model given input language model state and a specific
   * token, return a new language model state and score.
   */
  virtual LMStatePtr
  score(const LMStatePtr& inState, int tokenIdx, float& score) = 0;

  /* Query the language model and finish decoding. */
  virtual LMStatePtr finish(const LMStatePtr& inState, float& score) = 0;

  /* Compare two language model states. */
  virtual int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const = 0;

 protected:
  LM() = default;
  virtual ~LM() = default;
};

typedef std::shared_ptr<LM> LMPtr;

} // namespace w2l
