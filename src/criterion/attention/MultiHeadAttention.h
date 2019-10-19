/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "criterion/attention/AttentionBase.h"

namespace w2l {

class MultiHeadContentAttention : public AttentionBase {
 public:
  MultiHeadContentAttention() {}
  explicit MultiHeadContentAttention(
      int dim,
      int num_heads = 8,
      bool keyValue = false,
      bool splitInput = false);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  int numHeads_;
  bool keyValue_;
  bool splitInput_;
  FL_SAVE_LOAD_WITH_BASE(AttentionBase, numHeads_, keyValue_, splitInput_)
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::MultiHeadContentAttention)
