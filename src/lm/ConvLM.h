/**
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */
#pragma once

#include <flashlight/flashlight.h>

#include "common/Dictionary.h"
#include "decoder/LM.h"

namespace w2l {

struct ConvLMState {
  std::vector<int> tokens;
  int length;

  ConvLMState() : length(0) {}
  explicit ConvLMState(int size)
      : tokens(std::vector<int>(size)), length(size) {}
};

class ConvLM : public LM {
 public:
  ConvLM(
      const std::string& modelPath,
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

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override;

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
  std::shared_ptr<fl::Module> network_;

  int vocabSize_;
  int maxHistorySize_;

  std::vector<std::vector<float>> getLogProb(
      const std::vector<int>& input,
      const std::vector<int>& lastTokenPositions,
      int sampleSize = -1,
      int batchSize = 1);

  static ConvLMState* getRawState(const LMStatePtr& state);

  std::pair<LMStatePtr, float> scoreWithLmIdx(
      const LMStatePtr& state,
      const int tokenIdx);
};

} // namespace w2l
