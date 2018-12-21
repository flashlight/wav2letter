/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <mkl.h>

#include "FeatureParams.h"

namespace speech {

// Convert the speech signal into frames
template <typename T>
std::vector<T> frameSignal(
    const std::vector<T>& input,
    const FeatureParams& params);

// row major;  matA - m x k , matB - k x n
template <typename T>
extern std::vector<T> mklGemm(
    const std::vector<T>& matA,
    const std::vector<T>& matB,
    MKL_INT n,
    MKL_INT k);

} // namespace speech
