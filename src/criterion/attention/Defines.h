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

// attention
const std::string kContentAttention = "content";
const std::string kNeuralContentAttention = "neural";
// window
const std::string kMedianWindow = "median";
const std::string kNoWindow = "no";
const std::string kSoftWindow = "soft";
const std::string kStepWindow = "step";

} // namespace w2l
