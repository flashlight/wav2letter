/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/FlashlightUtils.h"

namespace w2l {

int64_t numTotalParams(std::shared_ptr<fl::Module> module) {
  int64_t params = 0;
  for (auto& p : module->params()) {
    params += p.elements();
  }
  return params;
}

af::array pad(
    const af::array& in,
    const int size,
    const int dim /* = 0 */,
    float val /* = 0.0 */) {
  if (size < 0) {
    throw std::invalid_argument(
        "Size must be non-negative. Given: " + std::to_string(size));
  }
  auto opdims = in.dims();
  opdims[dim] += (size << 1);
  af::array op = af::constant(val, opdims, in.type());

  std::array<af::seq, 4> sel = {af::span, af::span, af::span, af::span};
  sel[dim] = af::seq(size, opdims[dim] - size - 1);
  op(sel[0], sel[1], sel[2], sel[3]) = in;

  return op;
}

} // namespace w2l
