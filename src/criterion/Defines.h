/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>
#include <functional>

namespace w2l {

using CriterionScaleFn = std::function<float(
    int64_t /* alphabet size */,
    int64_t /* timeframes */,
    int64_t /* labelsize */)>;

enum class CriterionScaleMode {
  NONE = 0,
  INPUT_SZ = 1,
  INPUT_SZ_SQRT = 2,
  TARGET_SZ = 3,
  TARGET_SZ_SQRT = 4,
};

// sampling strategy to use in decoder in place of teacher forcing
const std::string kModelSampling = "model";
const std::string kRandSampling = "rand";
} // namespace w2l
