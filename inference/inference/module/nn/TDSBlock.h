/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cstdio>
#include <memory>

#include "inference/common/IOBuffer.h"
#include "inference/module/ModuleParameter.h"
#include "inference/module/nn/Conv1d.h"
#include "inference/module/nn/LayerNorm.h"
#include "inference/module/nn/Linear.h"
#include "inference/module/nn/Sequential.h"

namespace w2l {
namespace streaming {

class TDSBlock : public Sequential {
 public:
  explicit TDSBlock(
      std::shared_ptr<Conv1d> conv,
      std::shared_ptr<LayerNorm> layernorm1,
      std::shared_ptr<Linear> linear1,
      std::shared_ptr<Linear> linear2,
      std::shared_ptr<LayerNorm> layernorm2,
      DataType reluDataType,
      DataType residualDataType);

  virtual ~TDSBlock() override = default;

  std::string debugString() const override;

 protected:
  DataType reluDataType_;
  DataType residualDataType_;

 private:
  friend class cereal::access;

  TDSBlock(); // Used by Cereal for serialization.

  template <class Archive>
  void serialize(Archive& ar) {
    ar(cereal::base_class<Sequential>(this), reluDataType_, residualDataType_);
  }
};

} // namespace streaming
} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::streaming::TDSBlock);
