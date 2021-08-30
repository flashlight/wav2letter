/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace w2l {
namespace streaming {

void meanAndStddev(const float* in, int size, float& mean, float& stddev);

// out = bias + weight * (in - mean) / stddev
void meanNormalize(
    const float* in,
    int size,
    float mean,
    float stddev,
    float weight,
    float bias,
    float* output);

} // namespace streaming
} // namespace w2l
