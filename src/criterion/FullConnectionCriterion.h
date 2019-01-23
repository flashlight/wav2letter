/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>
#include "CriterionUtils.h"
#include "Defines.h"

namespace w2l {

class FullConnectionCriterion : public fl::BinaryModule {
 public:
  explicit FullConnectionCriterion(
      int N,
      w2l::CriterionScaleMode scalemode = w2l::CriterionScaleMode::NONE);

  fl::Variable forward(const fl::Variable& input, const fl::Variable& target)
      override;

  std::string prettyString() const override;

 private:
  friend class AutoSegmentationCriterion;
  FullConnectionCriterion() = default;

  int N_;
  w2l::CriterionScaleMode scaleMode_;

  FL_SAVE_LOAD_WITH_BASE(
      fl::BinaryModule,
      fl::serializeAs<int64_t>(N_),
      scaleMode_)

  struct fwParams {
    std::vector<int> targetsRaw;
    std::vector<float> inputsRaw, transRaw, scale;

    std::vector<float> res;
    std::vector<double> alpha;
    std::vector<int> alphaIndex;

    fwParams(int n, int t, int b, int l) {
      targetsRaw.resize(l * b);
      inputsRaw.resize(b * t * n);
      alphaIndex.resize(b * n * t);
      res.resize(b);
      scale.resize(b);
      alpha.resize(b * n * t);
      transRaw.resize(n * n);
    }
  };

  struct bwParams {
    std::vector<double> alphaGrad;
    std::vector<float> inputsGrad, transGradRes, outputsGrad;
    std::vector<double> transGrad;

    bwParams(int n, int t, int b) {
      alphaGrad.resize(b * n * t, 0);
      inputsGrad.resize(b * t * n, 0);
      transGrad.resize(b * n * n, 0);
      outputsGrad.resize(b);
      transGradRes.resize(n * n, 0);
    }
  };
};

typedef FullConnectionCriterion FCCLoss;

} // namespace w2l

CEREAL_REGISTER_TYPE(w2l::FullConnectionCriterion)
