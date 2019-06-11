/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Transforms.h"

#include <functional>

#include "common/Defines.h"
#include "common/Utils-base.h"

namespace w2l {

void replaceReplabels(
    std::vector<int>& in,
    int64_t numreps,
    const Dictionary& dict) {
  if (in.empty() || numreps == 0) {
    return;
  }

  std::unordered_map<int64_t, int64_t> repmap;
  for (int64_t i = 1; i <= numreps; ++i) {
    repmap[i] = dict.getIndex(std::to_string(i));
  }
  replaceReplabels(in, numreps, repmap);
}

void replaceReplabels(
    std::vector<int>& in,
    int64_t numreps,
    const std::unordered_map<int64_t, int64_t>& repmap) {
  if (in.empty() || numreps == 0) {
    return;
  }

  size_t ptr0 = 0, ptr1 = 0;
  while (ptr1 < in.size()) {
    in[ptr0++] = in[ptr1++];
    int64_t s = 0;
    while (s < numreps && ptr1 < in.size() && in[ptr1] == in[ptr1 - 1]) {
      ptr1++;
      s++;
    }
    if (s > 0) {
      auto sidx = repmap.find(s);
      in[ptr0++] = sidx->second;
    }
  }
  in.resize(ptr0);
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
