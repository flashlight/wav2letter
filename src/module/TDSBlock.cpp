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
    int c,
    int kw,
    int h,
    double dropout /* = 0 */,
    int l2 /* = 0 */) {
  Sequential conv;
  conv.add(Conv2D(c, c, kw, 1, 1, 1, -1, -1));
  conv.add(ReLU());
  conv.add(Dropout(dropout));

  int l = c * h;
  if (l2 == 0) {
    l2 = l;
  }
  Sequential fc;
  fc.add(View(af::dim4(-1, l, 1, 0)));
  fc.add(Reorder(1, 0, 2, 3));
  fc.add(Linear(l, l2));
  fc.add(ReLU());
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }
  fc.add(Linear(l2, l));
  fc.add(Reorder(1, 0, 2, 3));
  fc.add(View(af::dim4(-1, h, c, 0)));
  if (dropout > 0) {
    fc.add(Dropout(dropout));
  }

  add(conv);
  add(LayerNorm(3));
  add(fc);
  add(LayerNorm(3));
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
