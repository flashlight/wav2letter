/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <numeric>

namespace w2l {
namespace streaming {

void meanAndStddev(const float* in, int size, float& mean, float& stddev) {
  float sum = std::accumulate(in, in + size, 0.0);
  mean = sum / size;
  float sqSum = std::inner_product(in, in + size, in, 0.0);
  stddev = std::sqrt(sqSum / size - mean * mean);
}

void meanNormalize(
    const float* in,
    int size,
    float mean,
    float stddev,
    float weight,
    float bias,
    float* output) {
  std::transform(in, in + size, output, [mean, stddev, weight, bias](float x) {
    return bias + weight * ((x - mean) / stddev);
  });
}

} // namespace streaming
} // namespace w2l
