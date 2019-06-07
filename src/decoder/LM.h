/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace w2l {
/**
 * LMStatePtr is a shared void* tracking LM states generated during decoding.
 */
using LMStatePtr = std::shared_ptr<void>;

/**
 * LM is a thin wrapper for laguage models. We abstrct several common methods
 * here which can be shared for KenLM, ConvLM, RNNLM, etc.
 */
class LM {
 public:
  /* Initialize or reset language model */
  virtual LMStatePtr start(bool startWithNothing) = 0;

  /**
   * Query the language model given input language model state and a specific
   * token, return a new language model state and score.
   */
  virtual std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) = 0;

  /* Query the language model and finish decoding. */
  virtual std::pair<LMStatePtr, float> finish(const LMStatePtr& state) = 0;

  /* Compare two language model states. */
  virtual int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const = 0;

  /* Update LM caches (optional) given a bunch of new states generated */
  virtual void updateCache(std::vector<LMStatePtr> stateIdices) {}

 protected:
  LM() = default;
  virtual ~LM() = default;

  /* Map indices from acoustic model to LM for each valid token. */
  std::unordered_map<int, int> usrToLmIdxMap_;
};

typedef std::shared_ptr<LM> LMPtr;

} // namespace w2l
