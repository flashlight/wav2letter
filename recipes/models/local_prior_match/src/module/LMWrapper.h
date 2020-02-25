/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include <string>
#include <vector>

namespace w2l {

// a simple wrapper for LMs trained with a different dictionary
class LMWrapper : public fl::Container {
 public:
  LMWrapper(
      std::shared_ptr<fl::Module> network,
      const std::vector<int>& dictIndexMap,
      int startIndex);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  std::string prettyString() const override;

 private:
  af::array dictIndexMap_;
  int startIndex_;

  FL_SAVE_LOAD_WITH_BASE(Container, dictIndexMap_, startIndex_)

  LMWrapper() = default;

  std::shared_ptr<fl::Module> lm() const {
    return module(0);
  }
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::LMWrapper)
