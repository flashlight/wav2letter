/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <module/TDSBlock.h>

namespace w2l {

using namespace fl;

TDSBlock::TDSBlock(
    int channels,
    int kernelSize,
    int width,
    double dropout /* = 0 */,
    int innerLinearDim /* = 0 */,
    int rightPadding /* = -1 */,
    bool lNormIncludeTime /* = true */) {
  Sequential conv;
  auto convPadding = static_cast<int>(fl::PaddingMode::SAME);
  if (rightPadding != -1) {
    int totalPadding = kernelSize - 1;
    if (rightPadding > totalPadding) {
      throw std::invalid_argument(
          "right padding exceeds the 'SAME' padding required for TDSBlock");
    }
    conv.add(Padding(
        std::pair<int, int>{totalPadding - rightPadding, rightPadding}, 0.0));
    convPadding = 0;
  }
  conv.add(Conv2D(channels, channels, kernelSize, 1, 1, 1, convPadding, 0));
  conv.add(ReLU());
  conv.add(Dropout(dropout));

  int linearDim = channels * width;
  if (innerLinearDim == 0) {
    innerLinearDim = linearDim;
  }
  Sequential fc;
  fc.add(Reorder(2, 1, 0, 3));
  fc.add(View(af::dim4(linearDim, -1, 1, 0)));

  fc.add(Linear(linearDim, innerLinearDim));
  fc.add(ReLU());
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }
  fc.add(Linear(innerLinearDim, linearDim));
  fc.add(View(af::dim4(channels, width, -1, 0)));
  fc.add(Reorder(2, 1, 0, 3));
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }

  add(conv);
  if (lNormIncludeTime) {
    add(LayerNorm(std::vector<int>{0, 1, 2}));
  } else {
    add(LayerNorm(std::vector<int>{1, 2}));
  }
  add(fc);
  if (lNormIncludeTime) {
    add(LayerNorm(std::vector<int>{0, 1, 2}));
  } else {
    add(LayerNorm(std::vector<int>{1, 2}));
  }
}

std::vector<Variable> TDSBlock::forward(const std::vector<Variable>& inputs) {
  auto out = inputs[0];
  out = module(0)->forward({out})[0] + out;
  out = module(1)->forward({out})[0];
  out = module(2)->forward({out})[0] + out;
  return module(3)->forward({out});
}

std::string TDSBlock::prettyString() const {
  std::ostringstream ss;
  auto convW = param(0);
  auto linW = param(4);
  int kw = convW.dims(0);
  int c = convW.dims(2);
  int w = linW.dims(0) / c;
  int l = linW.dims(1);
  int l2 = linW.dims(0);
  ss << "Time-Depth Separable Block (";
  ss << kw << ", " << w << ", " << c << ") [" << l << " -> " << l2 << " -> "
     << l << "]";
  return ss.str();
}

} // namespace w2l
