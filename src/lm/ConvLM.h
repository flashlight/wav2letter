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
#include <glog/logging.h>
#include "common/Dictionary.h"
#include "decoder/LM.h"

namespace w2l {

struct ConvLMState : public LMState {
  std::vector<int> tokens;
  int length;

  ConvLMState() : length(0) {}
  explicit ConvLMState(int size)
      : tokens(std::vector<int>(size)), length(size) {}
};

class ConvLM : public LM {
 public:
  int index(const std::string& token) override;

  LMStatePtr start(bool startWithNonEos) override;

  LMStatePtr score(const LMStatePtr& inState, int tokenIdx, float& score)
      override;

  LMStatePtr finish(const LMStatePtr& inState, float& score) override;

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override;

  explicit ConvLM(
      const std::string& modelPath,
      const std::string& tokenVocabPath,
      int lmMemory = 10000,
      int beamSize = 2500,
      int historySize = 49);

  void updateCache(std::vector<LMStatePtr> states) override;

 private:
  // This cache is not thread-safe!
  int lmMemory_;
  int beamSize_;
  std::unordered_map<const ConvLMState*, int> cacheIndices_;
  std::vector<std::vector<float>> cache_;
  std::vector<const ConvLMState*> slot_;
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

  typedef std::shared_ptr<ConvLM> ConvLMPtr;
};

} // namespace w2l
