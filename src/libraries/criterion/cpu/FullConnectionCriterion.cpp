/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cpu/FullConnectionCriterion.h"

#include <cmath>

#include "libraries/common/Utils.h"
#include "libraries/common/Workspace.h"
#include "libraries/criterion/cpu/CriterionUtils.h"

namespace {

template <class Float>
struct WorkspacePtrs {
  explicit WorkspacePtrs(void* workspace, int B, int T, int N) {
    w2l::Workspace<> ws(workspace);
    ws.request(&scale, B);
    ws.request(&alpha, B, T, N);
    ws.request(&alphaGrad, B, T, N);
    ws.request(&transBatchGrad, B, N, N);
    ws.request(&transBuf, B, N, N);
    requiredSize = ws.requiredSize();
  }

  Float* scale;
  double* alpha;
  double* alphaGrad;
  double* transBatchGrad;
  double* transBuf;
  size_t requiredSize;
};

} // namespace

namespace w2l {
namespace cpu {

template <class Float>
size_t FullConnectionCriterion<Float>::getWorkspaceSize(int B, int T, int N) {
  return WorkspacePtrs<Float>(nullptr, B, T, N).requiredSize;
}

template <class Float>
void FullConnectionCriterion<Float>::forward(
    int B,
    int T,
    int N,
    CriterionScaleMode scaleMode,
    const Float* input,
    const int* targetSize,
    const Float* trans,
    Float* loss,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);
  CriterionUtils<Float>::computeScale(B, T, N, scaleMode, targetSize, ws.scale);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    for (int n = 0; n < N; ++n) {
      int k = b * T * N + n;
      ws.alpha[k] = input[k];
    }

    for (int t = 1; t <= T; ++t) {
      for (int m = 0; m < N; ++m) {
        const auto* alphaPrev = &ws.alpha[b * T * N + (t - 1) * N];
        const auto* inputCur = &input[b * T * N + t * N];
        auto* transBuf = &ws.transBuf[b * N * N + m * N];
        auto* alphaCur = &ws.alpha[b * T * N + t * N];

        double maxValue = -INFINITY;
        for (int n = 0; n < N; ++n) {
          double val = transBuf[n] =
              alphaPrev[n] + (t == T ? 0 : trans[m * N + n]);
          maxValue = val > maxValue ? val : maxValue;
        }

        double sumValue = 0;
        for (int n = 0; n < N; ++n) {
          sumValue += exp(transBuf[n] - maxValue);
        }

        if (t == T) {
          loss[b] = ws.scale[b] * (log(sumValue) + maxValue);
          break;
        }

        alphaCur[m] = log(sumValue) + maxValue + inputCur[m];
      }
    }
  }
}

template <class Float>
void FullConnectionCriterion<Float>::backward(
    int B,
    int T,
    int N,
    const Float* trans,
    const Float* grad,
    Float* _inputGrad,
    Float* transGrad,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N);
  setZero(_inputGrad, B * T * N);
  setZero(transGrad, N * N);
  setZero(ws.alphaGrad, B * T * N);
  setZero(ws.transBatchGrad, B * N * N);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    for (int t = T; t > 0; --t) {
      for (int m = 0; m < N; ++m) {
        const auto* alphaPrev = &ws.alpha[b * T * N + (t - 1) * N];
        const auto* alphaCurGrad = &ws.alphaGrad[b * T * N + t * N];
        auto* alphaPrevGrad = &ws.alphaGrad[b * T * N + (t - 1) * N];
        auto* transBuf = &ws.transBuf[b * N * N + m * N];
        auto* transBatchGrad = &ws.transBatchGrad[b * N * N + m * N];

        double maxValue = -INFINITY;
        for (int n = 0; n < N; ++n) {
          double val = transBuf[n] =
              alphaPrev[n] + (t == T ? 0 : trans[m * N + n]);
          maxValue = val > maxValue ? val : maxValue;
        }

        double sumValue = 0;
        for (int n = 0; n < N; ++n) {
          transBuf[n] = exp(transBuf[n] - maxValue);
          sumValue += transBuf[n];
        }

        if (t == T) {
          for (int n = 0; n < N; ++n) {
            alphaPrevGrad[n] = transBuf[n] / sumValue;
          }
          break;
        }

        for (int n = 0; n < N; ++n) {
          transBuf[n] = transBuf[n] / sumValue * alphaCurGrad[m];
          transBatchGrad[n] += transBuf[n];
        }
      }

      if (t == T) {
        continue;
      }

      for (int m = 0; m < N; ++m) {
        auto* alphaPrevGrad = &ws.alphaGrad[b * T * N + (t - 1) * N];

        for (int n = 0; n < N; ++n) {
          alphaPrevGrad[m] += ws.transBuf[b * N * N + n * N + m];
        }
      }
    }

    auto* alphaGrad = &ws.alphaGrad[b * T * N];
    auto* inputGrad = &_inputGrad[b * T * N];

    for (int i = 0; i < T * N; ++i) {
      inputGrad[i] = ws.scale[b] * grad[b] * alphaGrad[i];
    }
  }

  for (int b = 0; b < B; ++b) {
    auto* transBatchGrad = &ws.transBatchGrad[b * N * N];

    for (int i = 0; i < N * N; ++i) {
      transGrad[i] += ws.scale[b] * grad[b] * transBatchGrad[i];
    }
  }
}

template struct FullConnectionCriterion<float>;
template struct FullConnectionCriterion<double>;

} // namespace cpu
} // namespace w2l
