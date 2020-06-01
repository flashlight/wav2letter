/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <functional>

#include "libraries/common/Defines.h"
#include "libraries/common/Dictionary.h"
#include "libraries/lm/LM.h"

namespace w2l {

using GetConvLmScoreFunc = std::function<std::vector<
    float>(const std::vector<int>&, const std::vector<int>&, int, int)>;

struct ConvLMState : LMState {
  std::vector<int> tokens;
  int length;

  ConvLMState() : length(0) {}
  explicit ConvLMState(int size)
      : tokens(std::vector<int>(size)), length(size) {}
};

class ConvLM : public LM {
 public:
  ConvLM(
      const GetConvLmScoreFunc& getConvLmScoreFunc,
      const std::string& tokenVocabPath,
      const Dictionary& usrTknDict,
      int lmMemory = 10000,
      int beamSize = 2500,
      int historySize = 49);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;

  void updateCache(std::vector<LMStatePtr> states) override;

 private:
  // This cache is also not thread-safe!
  int lmMemory_;
  int beamSize_;
  std::unordered_map<ConvLMState*, int> cacheIndices_;
  std::vector<std::vector<float>> cache_;
  std::vector<ConvLMState*> slot_;
  std::vector<int> batchedTokens_;

  Dictionary vocab_;
  GetConvLmScoreFunc getConvLmScoreFunc_;

  int vocabSize_;
  int maxHistorySize_;

  std::pair<LMStatePtr, float> scoreWithLmIdx(
      const LMStatePtr& state,
      const int tokenIdx);
};

} // namespace w2l
