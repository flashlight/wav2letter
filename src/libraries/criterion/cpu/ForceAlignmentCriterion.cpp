/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cpu/ForceAlignmentCriterion.h"

#include <algorithm>
#include <cmath>

#include "libraries/common/Utils.h"
#include "libraries/common/Workspace.h"
#include "libraries/criterion/cpu/CriterionUtils.h"

namespace {

template <class Float>
struct WorkspacePtrs {
  WorkspacePtrs(void* workspace, int B, int T, int N, int L) {
    w2l::Workspace<> ws(workspace);
    ws.request(&scale, B);
    ws.request(&alpha, B, T, L);
    ws.request(&alphaGrad, B, T, L);
    ws.request(&transBatchGrad, B, N, N);
    ws.request(&transBuf1, B, L);
    ws.request(&transBuf2, B, L);
    ws.request(&transBufGrad1, B, L);
    ws.request(&transBufGrad2, B, L);
    requiredSize = ws.requiredSize();
  }

  Float* scale;
  double* alpha;
  double* alphaGrad;
  Float* transBatchGrad;
  Float* transBuf1;
  Float* transBuf2;
  Float* transBufGrad1;
  Float* transBufGrad2;
  size_t requiredSize;
};

} // namespace

namespace w2l {
namespace cpu {

template <class Float>
size_t
ForceAlignmentCriterion<Float>::getWorkspaceSize(int B, int T, int N, int L) {
  WorkspacePtrs<Float> dummy(nullptr, B, T, N, L);
  return dummy.requiredSize;
}

template <class Float>
void ForceAlignmentCriterion<Float>::forward(
    int B,
    int T,
    int N,
    int _L,
    CriterionScaleMode scaleMode,
    const Float* _input,
    const int* _target,
    const int* targetSize,
    const Float* trans,
    Float* loss,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N, _L);
  CriterionUtils<Float>::computeScale(B, T, N, scaleMode, targetSize, ws.scale);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    auto* alpha = &ws.alpha[b * T * _L];
    auto* input = &_input[b * T * N];
    auto* target = &_target[b * _L];
    auto* transBuf1 = &ws.transBuf1[b * _L];
    auto* transBuf2 = &ws.transBuf2[b * _L];
    int L = targetSize[b];

    alpha[0] = input[target[0]];

    for (int i = 0; i < L; ++i) {
      transBuf1[i] = trans[target[i] * N + target[i]];
      transBuf2[i] = i > 0 ? trans[target[i] * N + target[i - 1]] : 0;
    }

    for (int t = 1; t < T; ++t) {
      auto* inputCur = &input[t * N];
      auto* alphaPrev = &alpha[(t - 1) * L];
      auto* alphaCur = &alpha[t * L];

      int high = t < L ? t : L;
      int low = T - t < L ? L - (T - t) : 1;

      if (T - t >= L) {
        alphaCur[0] = alphaPrev[0] + transBuf1[0] + inputCur[target[0]];
      }

      if (t < L) {
        alphaCur[high] =
            alphaPrev[high - 1] + transBuf2[high] + inputCur[target[high]];
      }

      for (int i = low; i < high; ++i) {
        double s1 = alphaPrev[i] + transBuf1[i];
        double s2 = alphaPrev[i - 1] + transBuf2[i];
        // lse = logSumExp(s1, s2)
        double lse =
            s1 < s2 ? s2 + log1p(exp(s1 - s2)) : s1 + log1p(exp(s2 - s1));
        alphaCur[i] = lse + inputCur[target[i]];
      }
    }

    loss[b] = alpha[T * L - 1] * ws.scale[b];
  }
}

template <class Float>
void ForceAlignmentCriterion<Float>::backward(
    int B,
    int T,
    int N,
    int _L,
    const int* _target,
    const int* targetSize,
    const Float* grad,
    Float* _inputGrad,
    Float* transGrad,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N, _L);
  setZero(_inputGrad, B * T * N);
  setZero(transGrad, N * N);
  setZero(ws.alphaGrad, B * T * _L);
  setZero(ws.transBatchGrad, B * N * N);
  setZero(ws.transBufGrad1, B * _L);
  setZero(ws.transBufGrad2, B * _L);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    auto* alpha = &ws.alpha[b * T * _L];
    auto* alphaGrad = &ws.alphaGrad[b * T * _L];
    auto* inputGrad = &_inputGrad[b * T * N];
    auto* target = &_target[b * _L];
    auto* transBatchGrad = &ws.transBatchGrad[b * N * N];
    auto* transBuf1 = &ws.transBuf1[b * _L];
    auto* transBuf2 = &ws.transBuf2[b * _L];
    auto* transBufGrad1 = &ws.transBufGrad1[b * _L];
    auto* transBufGrad2 = &ws.transBufGrad2[b * _L];
    int L = targetSize[b];

    alphaGrad[T * L - 1] = 1;

    for (int t = T - 1; t > 0; --t) {
      auto* inputCurGrad = &inputGrad[t * N];
      auto* alphaPrev = &alpha[(t - 1) * L];
      auto* alphaCurGrad = &alphaGrad[t * L];
      auto* alphaPrevGrad = &alphaGrad[(t - 1) * L];

      int high = t < L ? t : L;
      int low = T - t < L ? L - (T - t) : 1;

      int high1 = t < L ? t + 1 : L;
      int low1 = T - t < L ? L - (T - t) : 0;

      for (int i = low1; i < high1; ++i) {
        inputCurGrad[target[i]] += alphaCurGrad[i];
      }

      if (T - t >= L) {
        alphaPrevGrad[0] += alphaCurGrad[0];
        transBufGrad1[0] += alphaCurGrad[0];
      }

      if (t < L) {
        alphaPrevGrad[high - 1] += alphaCurGrad[high];
        transBufGrad2[high] += alphaCurGrad[high];
      }

      for (int i = low; i < high; ++i) {
        double s1 = alphaPrev[i] + transBuf1[i];
        double s2 = alphaPrev[i - 1] + transBuf2[i];
        // d1, d2 = dLogSumExp(s1, s2)
        double d1, d2;
        if (s1 < s2) {
          d2 = 1 / (1 + exp(s1 - s2));
          d1 = 1 - d2;
        } else {
          d1 = 1 / (1 + exp(s2 - s1));
          d2 = 1 - d1;
        }
        alphaPrevGrad[i] += d1 * alphaCurGrad[i];
        alphaPrevGrad[i - 1] += d2 * alphaCurGrad[i];
        transBufGrad1[i] += d1 * alphaCurGrad[i];
        transBufGrad2[i] += d2 * alphaCurGrad[i];
      }
    }

    inputGrad[target[0]] += alphaGrad[0];
    auto gradScale = grad[b] * ws.scale[b];
    for (int i = 0; i < T * N; ++i) {
      inputGrad[i] *= gradScale;
    }

    for (int i = 0; i < L; ++i) {
      transBatchGrad[target[i] * N + target[i]] += transBufGrad1[i];
      if (i > 0) {
        transBatchGrad[target[i] * N + target[i - 1]] += transBufGrad2[i];
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    auto transBatchGrad = ws.transBatchGrad + b * N * N;
    auto gradScale = grad[b] * ws.scale[b];
    for (int i = 0; i < N * N; ++i) {
      transGrad[i] += gradScale * transBatchGrad[i];
    }
  }
}

template <class Float>
void ForceAlignmentCriterion<Float>::viterbi(
    int B,
    int T,
    int N,
    int _L,
    const Float* _input,
    const int* _target,
    const int* targetSize,
    const Float* trans,
    int* bestPaths,
    void* workspace) {
  WorkspacePtrs<Float> ws(workspace, B, T, N, _L);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; ++b) {
    double* alpha = &ws.alpha[b * T * _L];
    const Float* input = &_input[b * T * N];
    const int* target = &_target[b * _L];
    Float* transBuf1 = &ws.transBuf1[b * _L];
    Float* transBuf2 = &ws.transBuf2[b * _L];
    int L = targetSize[b];
    for (int i = 0; i < L * T; i++) {
      alpha[i] = -std::numeric_limits<Float>::infinity();
    }

    alpha[0] = input[target[0]];

    for (int i = 0; i < L; ++i) {
      transBuf1[i] = trans[target[i] * N + target[i]];
      transBuf2[i] = i > 0 ? trans[target[i] * N + target[i - 1]] : 0;
    }

    for (int t = 1; t < T; ++t) {
      const Float* inputCur = &input[t * N];
      double* alphaPrev = &alpha[(t - 1) * L];
      double* alphaCur = &alpha[t * L];

      int high = t < L ? t : L;
      int low = T - t < L ? L - (T - t) : 1;

      // Handle edge cases.
      // If (T - t >= L), then we can conceivably still be at the initial blank
      if (T - t >= L) {
        alphaCur[0] = alphaPrev[0] + transBuf1[0] + inputCur[target[0]];
      }

      // If (t < L), then the highest position can only be be computed
      // by transitioning. (We couldn't have been at position `high`
      // at the previous timestep).
      if (t < L) {
        alphaCur[high] =
            alphaPrev[high - 1] + transBuf2[high] + inputCur[target[high]];
      }

      for (int i = low; i < high; ++i) {
        double s1 = alphaPrev[i] + transBuf1[i];
        double s2 = alphaPrev[i - 1] + transBuf2[i];
        alphaCur[i] = inputCur[target[i]] + fmax(s1, s2);
      }
    }

    auto ltrIdx = L - 1;
    int* bestPath = bestPaths + b * T;
    for (auto t = T - 1; t > 0; t--) {
      bestPath[t] = target[ltrIdx];
      auto* alphaPrev = &alpha[(t - 1) * L];
      if (ltrIdx > 0) {
        double s1 = alphaPrev[ltrIdx] + transBuf1[ltrIdx];
        double s2 = alphaPrev[ltrIdx - 1] + transBuf2[ltrIdx];
        if (s2 > s1) {
          ltrIdx--;
        }
      }
    }
    bestPath[0] = target[ltrIdx];
  }
}

template struct ForceAlignmentCriterion<float>;
template struct ForceAlignmentCriterion<double>;

} // namespace cpu
} // namespace w2l
