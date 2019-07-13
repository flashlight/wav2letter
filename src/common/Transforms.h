/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "common/Defines.h"
#include "libraries/common/Dictionary.h"

namespace w2l {

template <class T>
void uniq(std::vector<T>& in) {
  if (in.empty()) {
    return;
  }
  auto it = std::unique(in.begin(), in.end());
  in.resize(std::distance(in.begin(), it));
}

template <class T>
void remapLabels(std::vector<T>& labels, const Dictionary& dict) {
  if (FLAGS_eostoken) {
    int eosidx = dict.getIndex(kEosToken);
    while (!labels.empty() && labels.back() == eosidx) {
      labels.pop_back();
    }
  }
  if (FLAGS_replabel > 0) {
    labels = unpackReplabels(labels, dict, FLAGS_replabel);
  }
  auto trimLabels = [&labels](int idx) {
    if (!labels.empty() && labels.back() == idx) {
      labels.pop_back();
    }
    if (!labels.empty() && labels.front() == idx) {
      labels.erase(labels.begin());
    }
  };
  if (dict.contains(kSilToken)) {
    trimLabels(dict.getIndex(kSilToken));
  }
  if (!FLAGS_surround.empty()) {
    trimLabels(dict.getIndex(FLAGS_surround));
  }
};

// Input: B x inRow x inCol (Row Major), Output: B x inCol x inRow (Row Major)
template <typename T>
std::vector<T> transpose2d(
    const std::vector<T>& in,
    int64_t inRow,
    int64_t inCol,
    int64_t inBatch = 1) {
  if (in.size() != inRow * inCol * inBatch) {
    throw std::invalid_argument("Invalid input size");
  }
  std::vector<T> out(in.size());
  for (size_t b = 0; b < inBatch; ++b) {
    int64_t start = b * inRow * inCol;
    for (size_t c = 0; c < inCol; ++c) {
      for (size_t r = 0; r < inRow; ++r) {
        out[start + c * inRow + r] = in[start + r * inCol + c];
      }
    }
  }
  return out;
}

template <typename T>
std::vector<T> localNormalize(
    const std::vector<T>& in,
    int64_t leftCtxSize,
    int64_t rightCtxSize,
    int64_t frameSz = 1,
    int64_t batchSz = 1,
    double threshold = 0.0) {
  if (in.empty()) {
    return {};
  }
  int64_t perBatchSz = in.size() / batchSz;
  int64_t perFrameSz = perBatchSz / frameSz;
  auto out(in);
  for (size_t b = 0; b < batchSz; ++b) {
    std::vector<T> sum(frameSz, 0.0), sum2(frameSz, 0.0);
    int64_t curFrame = 0;
    // accumulate sum, sum^2 for computing mean, stddev
    for (auto i = b * perBatchSz; i < (b + 1) * perBatchSz; ++i) {
      auto start = std::max(curFrame - rightCtxSize, 0L);
      auto end = std::min(curFrame + leftCtxSize, frameSz - 1);
      for (int64_t j = start; j <= end; ++j) {
        sum[j] += in[i];
        sum2[j] += in[i] * in[i];
      }
      curFrame = (curFrame + 1) % frameSz;
    }
    // compute mean, stddev
    for (auto j = 0; j < frameSz; ++j) {
      int64_t N = (std::min(j + rightCtxSize, frameSz - 1) -
                   std::max(j - leftCtxSize, 0L) + 1) *
          perFrameSz;
      sum[j] /= N;
      sum2[j] /= N;
      sum2[j] -= (sum[j] * sum[j]);
      sum2[j] = std::sqrt(sum2[j]);
    }
    // perform local normalization
    curFrame = 0;
    for (auto i = b * perBatchSz; i < (b + 1) * perBatchSz; ++i) {
      out[i] -= sum[curFrame];
      if (sum2[curFrame] > threshold) {
        out[i] /= sum2[curFrame];
      }
      curFrame = (curFrame + 1) % frameSz;
    }
  }
  return out;
}

template <typename T>
std::vector<T> normalize(
    const std::vector<T>& in,
    int64_t batchSz = 1,
    double threshold = 0.0) {
  if (in.empty()) {
    return {};
  }
  auto out(in);
  int64_t perBatchSz = out.size() / batchSz;
  for (size_t b = 0; b < batchSz; ++b) {
    auto start = out.begin() + b * perBatchSz;
    T sum = std::accumulate(start, start + perBatchSz, 0.0);
    T mean = sum / perBatchSz;
    std::transform(
        start, start + perBatchSz, start, [mean](T x) { return x - mean; });
    T sq_sum = std::inner_product(start, start + perBatchSz, start, 0.0);
    T stddev = std::sqrt(sq_sum / perBatchSz);
    if (stddev > threshold) {
      std::transform(start, start + perBatchSz, start, [stddev](T x) {
        return x / stddev;
      });
    }
  }
  return out;
}
} // namespace w2l
