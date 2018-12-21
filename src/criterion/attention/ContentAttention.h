/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "AttentionBase.h"

namespace fl {

class ContentAttention : public AttentionBase {
 public:
  ContentAttention() {}

  std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class NeuralContentAttention : public AttentionBase {
 public:
  NeuralContentAttention() {}
  explicit NeuralContentAttention(int dim, int layers = 1);

  std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::ContentAttention)
CEREAL_REGISTER_TYPE(fl::NeuralContentAttention)
