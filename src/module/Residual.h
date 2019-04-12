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
 * A module for creating a generic residual block. Currently supports
 * skip connection multiplied by scale factor: (x + f(x)) * scale.
 * Connection from layer-0 means take the input to residual block
 * Connection to the layer-(N_layers + 1) means add to the output of residual
 * block.
 * For example, if a residual block has 4 layers, layer-1, layer-2, layer-3, and
 * layer-4: addShortcut(fromLayer, toLayer) - function to add skip connection
 * (1) addShortcut(0, 3) adds a skip connection from the input
 *     to layer-3 in the residual block:
 *        input(layer-3) = input(residual block) + output(layer-2)
 * (2) addShortcut(2, 5) adds a skip connection from layer-2 to the final output
          output(residual block) = output(layer-2) + output(layer-4)
 * To scale the sum of all inputs to the layer-k
 * one can use funtion addScale(layer_k, scale)
 * For instance, in the above example to scale the output of residual block
 * by factor 2.5, we can call addScale(5, 2.5).
 */
class Residual : public fl::Container {
 private:
  void checkShortcut(int fromLayer, int toLayer);
  void processShortcut(int fromLayer, int toLayer, int projectionIndex);
  fl::Variable applyScale(const fl::Variable& input, const int layerIndex);

  std::unordered_map<int, std::unordered_map<int, int>>
      shortcut_; // end -> start
  std::unordered_set<int> projectionsIndices_;
  std::unordered_map<int, float> scales_;

  FL_SAVE_LOAD_WITH_BASE(fl::Container, shortcut_, scales_, projectionsIndices_)

 public:
  Residual() = default;

  void addScale(int beforeLayer, float scale);

  void addShortcut(int fromLayer, int toLayer);

  template <typename T>
  void addShortcut(int fromLayer, int toLayer, const T& module) {
    // fromLayer: 0, .., nLayers_ - 1; toLayer: 1, 2, .., nLayers_ + 1
    // toLayer - fromLayer > 1 (avoid adding skip connection
    // from layer-K to layer-K+1)
    // add also layer applied to the output of fromLayer
    // to have the same dimensions as input to toLayer
    addShortcut(fromLayer, toLayer, std::make_shared<T>(module));
  }

  template <typename T>
  void addShortcut(int fromLayer, int toLayer, std::shared_ptr<T> module) {
    // fromLayer: 0, .., nLayers_ - 1; toLayer: 1, 2, .., nLayers_ + 1
    // toLayer - fromLayer > 1 (avoid adding skip connection
    // from layer-K to layer-K+1)
    // add also layer applied to the output of fromLayer
    // to have the same dimensions as input to toLayer
    checkShortcut(fromLayer, toLayer);
    Container::add(module);
    processShortcut(fromLayer, toLayer, modules_.size() - 1);
    projectionsIndices_.insert(modules_.size() - 1);
  }

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;

  fl::Variable forward(const fl::Variable& input);

  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::Residual)
