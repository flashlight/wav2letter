/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace w2l {

class AttentionBase : public fl::Container {
 public:
  AttentionBase() {}

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override {
    if (inputs.size() != 3 && inputs.size() != 4) {
      throw std::invalid_argument("Invalid inputs size");
    }

    auto attnWeight = inputs.size() == 4 ? inputs[3] : fl::Variable();
    auto res = forward(inputs[0], inputs[1], inputs[2], attnWeight);
    return {res.first, res.second};
  }

  std::pair<fl::Variable, fl::Variable> operator()(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn) {
    return forward(state, xEncoded, prevAttn);
  }

  std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn) {
    return forward(state, xEncoded, prevAttn, fl::Variable() /* attnWeight */);
  }

  std::pair<fl::Variable, fl::Variable> operator()(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) {
    return forward(state, xEncoded, prevAttn, attnWeight);
  }

  virtual std::pair<fl::Variable, fl::Variable> forward(
      const fl::Variable& state,
      const fl::Variable& xEncoded,
      const fl::Variable& prevAttn,
      const fl::Variable& attnWeight) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(fl::Container)
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::AttentionBase)
