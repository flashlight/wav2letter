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

class TokenLMDecoder : public LexiconDecoder {
 public:
  TokenLMDecoder(
      const DecoderOptions& opt,
      const TriePtr lexicon,
      const LMPtr lm,
      const int sil,
      const int blank,
      const TrieLabelPtr unk,
      const std::vector<float>& transitions,
      const std::unordered_map<int, int>& lmIndMap)
      : LexiconDecoder(opt, lexicon, lm, sil, blank, unk, transitions),
        lmIndMap_(lmIndMap) {}

  void decodeStep(const float* emissions, int T, int N) override;

 protected:
  int mergeCandidates(const int size) override;

  std::unordered_map<int, int> lmIndMap_;
};

} // namespace w2l
