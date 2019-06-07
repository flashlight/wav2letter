/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cpu/ViterbiPath.h"

#include <cmath>

#include "libraries/common/Workspace.h"

namespace {

template <class Float>
struct WorkspacePtrs {
  explicit WorkspacePtrs(void* workspace, int B, int T, int N) {
    w2l::Workspace<> ws(workspace);
    ws.request(&alpha, B, 2, N);
    ws.request(&beta, B, T, N);
    requiredSize = ws.requiredSize();
  }

  Float* alpha;
  int* beta;
  size_t requiredSize;
};

} // namespace

namespace w2l {
namespace cpu {

template <class Float>
size_t ViterbiPath<Float>::getWorkspaceSize(int B, int T, int N) {
  return WorkspacePtrs<Float>(nullptr, B, T, N).requiredSize;
}

template <class Float>
void ViterbiPath<Float>::compute(
    int B,
    int T,
    int N,
    const Float* input,
    const Float* trans,
    int* _path,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < N; ++n) {
      ws.alpha[b * 2 * N + n] = input[b * T * N + n];
    }

    for (int t = 1; t <= T; ++t) {
      const auto* alphaPrev = &ws.alpha[b * 2 * N + ((t - 1) % 2) * N];
      const auto* inputCur = &input[b * T * N + t * N];
      auto* alphaCur = &ws.alpha[b * 2 * N + (t % 2) * N];
      auto* betaCur = &ws.beta[b * T * N + t * N];

      for (int m = 0; m < N; ++m) {
        int maxIndex = -1;
        Float maxValue = -INFINITY;
        for (int n = 0; n < N; ++n) {
          Float val = alphaPrev[n] + (t == T ? 0 : trans[m * N + n]);
          if (val > maxValue) {
            maxIndex = n;
            maxValue = val;
          }
        }

        if (t == T) {
          auto* path = &_path[b * T];
          path[T - 1] = maxIndex;
          for (int s = T - 1; s > 0; --s) {
            path[s - 1] = ws.beta[b * T * N + s * N + path[s]];
          }
          break;
        }

        alphaCur[m] = maxValue + inputCur[m];
        betaCur[m] = maxIndex;
      }
    }
  }
}

template struct ViterbiPath<float>;
template struct ViterbiPath<double>;

} // namespace cpu
} // namespace w2l
