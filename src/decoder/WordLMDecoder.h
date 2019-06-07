/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include "decoder/LM.h"
#include "decoder/LexiconDecoder.h"
#include "decoder/Trie.h"

namespace w2l {

class WordLMDecoder : public LexiconDecoder {
 public:
  WordLMDecoder(
      const DecoderOptions& opt,
      const TriePtr& lexicon,
      const LMPtr& lm,
      const int sil,
      const int blank,
      const int unk,
      const std::vector<float>& transitions)
      : LexiconDecoder(opt, lexicon, lm, sil, blank, unk, transitions) {}

  void decodeStep(const float* emissions, int T, int N) override;

 protected:
  void mergeCandidates() override;
};

} // namespace w2l
