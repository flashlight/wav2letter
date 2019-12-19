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

/**
 * Implements Time-Depth Separable Convolution Block as described in the paper
 * [Sequence-to-Sequence Speech Recognition with Time-Depth Separable
 * Convolutions](https://arxiv.org/abs/1904.02619).
 * This [link](https://imgur.com/a/LAdlwZK) shows the diagram of TDSBlock.
 */

class TDSBlock : public fl::Container {
 private:
  TDSBlock() = default;
  FL_SAVE_LOAD_WITH_BASE(fl::Container)

 public:
  /**
   * Constructs a TDS Block. Input/Output Dim: T x W x C x B, where
   *    T = number of time-steps
   *    W = input/output width
   *    C = input/output channels
   *    B = batch size
   *
   * @param channels Number of input (and output) channels
   * @param kernelSize Kernel size for convolution
   * @param width Input width
   * @param dropout Amount of dropout to be used
   * @param innerLinearDim If > 0, the two linear layers perform the `input`->
      `output` channel transform as `linearDim` -> `innerLinearDim` and
      `innerLinearDim` -> `linearDim`, where `linearDim` = `W * C`
   * @param rightPadding Amount of right padding for asymmetric convolutions.
      By default (= `-1`), performs symmetric padding for the "SAME" conv.
   * @param lNormIncludeTime If `true`, normalization is performed on the
      [T, W, C] axes as in the original paper. If `false`, exclude time
      dimension from computing stats to follow standard LayerNorm ==>
      normalization is performed on the [W, C] axes with scalar affine
      transformation.
   */
  explicit TDSBlock(
      int channels,
      int kernelSize,
      int width,
      double dropout = 0,
      int innerLinearDim = 0,
      int rightPadding = -1,
      bool lNormIncludeTime = true);

  std::vector<fl::Variable> forward(
      const std::vector<fl::Variable>& inputs) override;
  std::string prettyString() const override;
};

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::TDSBlock)
