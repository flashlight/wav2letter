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

class TDSBlock : public fl::Container {
 private:
  TDSBlock() = default;
  FL_SAVE_LOAD_WITH_BASE(fl::Container)

 public:
  explicit TDSBlock(int c, int kw, int h, double dropout = 0);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;
  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::TDSBlock)
