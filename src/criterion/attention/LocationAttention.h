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

class SimpleLocationAttention : public AttentionBase {
 public:
  explicit SimpleLocationAttention(int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  SimpleLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class LocationAttention : public AttentionBase {
 public:
  LocationAttention(int encDim, int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  LocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};

class NeuralLocationAttention : public AttentionBase {
 public:
  NeuralLocationAttention(
      int encDim,
      int attnDim,
      int convChannel,
      int convKernel);

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) override;

  std::string prettyString() const override;

 private:
  NeuralLocationAttention() = default;

  FL_SAVE_LOAD_WITH_BASE(AttentionBase)
};
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::SimpleLocationAttention)
CEREAL_REGISTER_TYPE(w2l::LocationAttention)
CEREAL_REGISTER_TYPE(w2l::NeuralLocationAttention)
