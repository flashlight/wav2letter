/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "FeatureParams.h"

namespace w2l {

// Convert the speech signal into frames

std::vector<float> frameSignal(
    const std::vector<float>& input,
    const FeatureParams& params);

// row major;  matA - m x k , matB - k x n

std::vector<float> cblasGemm(
    const std::vector<float>& matA,
    const std::vector<float>& matB,
    int n,
    int k);

} // namespace w2l
