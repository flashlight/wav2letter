/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

namespace fl {

class AttentionBase : public Container {
 public:
  AttentionBase() {}

  Variable forward(const Variable& input) {
    throw af::exception(
        "Attention module requires three inputs (state, encodex, prevAttn) for forward");
  }

  std::pair<Variable, Variable> operator()(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn) {
    return forward(state, xEncoded, prevAttn);
  }

  std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn) {
    return forward(state, xEncoded, prevAttn, Variable() /* attnWeight */);
  }

  virtual std::pair<Variable, Variable> forward(
      const Variable& state,
      const Variable& xEncoded,
      const Variable& prevAttn,
      const Variable& attnWeight) = 0;

 private:
  FL_SAVE_LOAD_WITH_BASE(Container)
};

} // namespace fl

CEREAL_REGISTER_TYPE(fl::AttentionBase)
