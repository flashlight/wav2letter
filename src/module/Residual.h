/*******************************************************
 * Copyright (c) 2017, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/
#pragma once

#include <cereal/types/set.hpp>
#include <cereal/types/unordered_set.hpp>

#include <flashlight/flashlight.h>

namespace w2l {

/**
 * A module for creating a generic residual block. Currently supports identity
 * skip connection. For example, if a residual block has 4 layers,
 * layer1, layer2, layer3, and layer4:
 * (1) addShortcut(0, 3) adds a skip connection from the input to the residual
 * block to layer3,
 * (2) addShortcut(2, 5) adds a skip connection from layer2 to the final output
 */
class Residual : public fl::Container {
 private:
  Residual() = default; // Intentionally private

  std::vector<std::unordered_set<int>> shortcut_; // start -> end
  std::vector<std::set<int>> reverseShortcut_; // end -> start

  FL_SAVE_LOAD_WITH_BASE(fl::Container, shortcut_, reverseShortcut_)

 public:
  explicit Residual(int num_layers);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  fl::Variable forward(const fl::Variable& input);

  void addShortcut(int startId, int endId);

  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::Residual)
