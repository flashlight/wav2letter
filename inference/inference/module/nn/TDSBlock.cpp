/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/TDSBlock.h"

#include <cassert>
#include <sstream>
#include <stdexcept>

#include "inference/module/nn/Relu.h"
#include "inference/module/nn/Residual.h"

namespace w2l {
namespace streaming {

TDSBlock::TDSBlock(
    std::shared_ptr<Conv1d> conv,
    std::shared_ptr<LayerNorm> layernorm1,
    std::shared_ptr<Linear> linear1,
    std::shared_ptr<Linear> linear2,
    std::shared_ptr<LayerNorm> layernorm2,
    DataType reluDataType,
    DataType residualDataType)
    : reluDataType_(reluDataType), residualDataType_(residualDataType) {
  auto convSeq = std::make_shared<Sequential>();
  convSeq->add(conv);
  convSeq->add(std::make_shared<Relu>(reluDataType_));
  add(std::make_shared<Residual>(convSeq, residualDataType_));
  add(layernorm1);

  auto linearSeq = std::make_shared<Sequential>();
  linearSeq->add(linear1);
  linearSeq->add(std::make_shared<Relu>(reluDataType_));
  linearSeq->add(linear2);

  add(std::make_shared<Residual>(linearSeq, residualDataType_));
  add(layernorm2);
}

TDSBlock::TDSBlock()
    : TDSBlock(
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          nullptr,
          DataType::UNINITIALIZED,
          DataType::UNINITIALIZED) {}

std::string TDSBlock::debugString() const {
  std::stringstream ss;
  ss << "TDSBlock: { \n";
  ss << Sequential::debugString() << "\n";
  ss << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
