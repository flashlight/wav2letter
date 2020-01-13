/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/Conv1d.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

namespace w2l {
namespace streaming {

Conv1d::Conv1d(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    int rightPadding,
    int leftPadding,
    int groups)
    : inChannels_(inChannels),
      outChannels_(outChannels),
      kernelSize_(kernelSize),
      stride_(stride),
      rightPadding_(rightPadding),
      leftPadding_(leftPadding),
      groups_(groups) {
  if (groups <= 0 || inChannels <= 0 || outChannels <= 0 || kernelSize <= 0 ||
      stride <= 0 || leftPadding < 0 || rightPadding < 0 ||
      // inChannels and outChannels must both be divisible by groups.
      (inChannels % groups) || (outChannels % groups)) {
    std::stringstream ss;
    ss << "Invalid argument at Conv1d::Conv1d("
       << "inChannels=" << inChannels << " outChannels=" << outChannels
       << " kernelSize=" << kernelSize << " stride=" << stride
       << " rightPadding=" << rightPadding << " leftPadding_=" << leftPadding
       << " groups=" << groups << ")";
    throw std::invalid_argument(ss.str());
  }
}

std::string Conv1d::debugString() const {
  std::stringstream ss;
  ss << "Conv1d:{inChannels_=" << inChannels_
     << " outChannels_=" << outChannels_ << " kernelSize_=" << kernelSize_
     << " stride_=" << stride_ << " rightPadding_=" << rightPadding_
     << " leftPadding_=" << leftPadding_ << " groups_=" << groups_ << " }";
  return ss.str();
}

std::shared_ptr<Conv1d> createConv1d(
    int inChannels,
    int outChannels,
    int kernelSize,
    int stride,
    int padding,
    int groups,
    std::shared_ptr<ModuleParameter> weights,
    std::shared_ptr<ModuleParameter> bias) {
  return createConv1d(
      inChannels,
      outChannels,
      kernelSize,
      stride,
      std::make_pair(padding, padding),
      groups,
      weights,
      bias);
}

} // namespace streaming
} // namespace w2l
