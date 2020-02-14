/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/criterion/cpu/ConnectionistTemporalClassificationCriterion.h"

#include <cmath>
#include <limits>

#include "libraries/common/Workspace.h"

namespace {

template <class Float>
struct WorkspacePtrs {
  WorkspacePtrs(void* workspace, int B, int T, int /* N unused */, int L) {
    const int s = (2 * L) + 1;
    w2l::Workspace<> ws(workspace);
    ws.request(&alpha, B, T, s);
    ws.request(&s_inc, B, s);
    ws.request(&e_inc, B, s);
    ws.request(&backptr, B, T, s);
    ws.request(&labels_w_blanks, B, s);
    requiredSize = ws.requiredSize();
  }

  Float* alpha;
  int* s_inc;
  int* e_inc;
  int* labels_w_blanks;
  int* backptr;
  size_t requiredSize;
};

/*
 * Derived from warpctc/include/detail/cpu_ctc.h
 * Creates labels_w_blanks which adds blank_labels between each character in
 * labels.
 * s_inc and e_inc are used by the `compute_alphas` kernel to determine the
 * furthest starting and end label location that each time step could possibly
 * be.
 */
int setup_labels(
    const int* labels,
    int* s_inc,
    int* e_inc,
    int* labels_w_blanks,
    int blank_label,
    int L,
    int S) {
  int e_counter = 0;
  int s_counter = 0;

  s_inc[s_counter++] = 1;

  int repeats = 0;

  for (int i = 1; i < L; ++i) {
    if (labels[i - 1] == labels[i]) {
      s_inc[s_counter++] = 1;
      s_inc[s_counter++] = 1;
      e_inc[e_counter++] = 1;
      e_inc[e_counter++] = 1;
      ++repeats;
    } else {
      s_inc[s_counter++] = 2;
      e_inc[e_counter++] = 2;
    }
  }
  e_inc[e_counter++] = 1;

  for (int i = 0; i < L; ++i) {
    labels_w_blanks[2 * i] = blank_label;
    labels_w_blanks[2 * i + 1] = labels[i];
  }
  labels_w_blanks[S - 1] = blank_label;

  return repeats;
}

/*
 * Derived from warpctc/include/detail/cpu_ctc.h
 * Float can be either float or double
 */
template <class Float>
void compute_alphas(
    const Float* input,
    int repeats,
    int S,
    int T,
    int N,
    const int* const e_inc,
    const int* const s_inc,
    const int* const labels,
    Float* alphas,
    int* backptr,
    int* paths) {
  const int blank_label_idx = N - 1;
  int start = (((S / 2) + repeats - T) < 0) ? 0 : 1, end = S > 1 ? 2 : 1;

  for (int i = 0; i < S * T; i++) {
    alphas[i] = -std::numeric_limits<Float>::infinity();
  }

  for (int i = start; i < end; ++i) {
    alphas[i] = input[labels[i]];
  }

  // Iterate through each time frame
  for (int t = 1; t < T; ++t) {
    // Calculate the smallest and largest possible index of the target that this
    // time could be
    int remain = (S / 2) + repeats - (T - t);
    if (remain >= 0) {
      start += s_inc[remain];
    }
    if (t <= (S / 2) + repeats) {
      end += e_inc[t - 1];
    }
    int startloop = start;
    int idx1 = t * S, idx2 = (t - 1) * S, idx3 = t * N;

    if (start == 0) {
      alphas[idx1] = alphas[idx2] + input[blank_label_idx + idx3];
      backptr[idx1] = 0;
      startloop += 1;
    }

    for (int i = startloop; i < end; ++i) {
      Float x0 = alphas[i + idx2];
      Float x1 = alphas[(i - 1) + idx2];
      Float x2 = -std::numeric_limits<Float>::infinity();

      // In CTC, the optimal path may optionally chose to skip a blank label.
      // x2 represents skipping a letter, and can only happen if we're not
      // currently on a blank_label, and we're not on a repeat letter
      // (i != 1) just ensures we don't access labels[i - 2] if its i < 2
      if (labels[i] != blank_label_idx && i != 1 &&
          labels[i] != labels[i - 2]) {
        x2 = alphas[(i - 2) + idx2];
      }
      Float result = 0.0;
      if (x2 > x1 && x2 > x0) {
        result = x2;
        backptr[i + idx1] = 2;
      } else if (x1 > x0 && x1 > x2) {
        result = x1;
        backptr[i + idx1] = 1;
      } else {
        result = x0;
        backptr[i + idx1] = 0;
      }
      alphas[i + idx1] = result + input[labels[i] + idx3];
    }
  }

  int ltrIdx = alphas[T * S - 1] > alphas[T * S - 2] ? S - 1 : S - 2;
  for (int t = T - 1; t >= 0; t--) {
    paths[t] = labels[ltrIdx];
    ltrIdx -= backptr[(t * S) + ltrIdx];
  }
}

} // namespace

namespace w2l {
namespace cpu {

template <class Float>
size_t ConnectionistTemporalClassificationCriterion<Float>::getWorkspaceSize(
    int B,
    int T,
    int N,
    int L) {
  WorkspacePtrs<Float> dummy(nullptr, B, T, N, L);
  return dummy.requiredSize;
}

template <class Float>
void ConnectionistTemporalClassificationCriterion<Float>::viterbi(
    int B,
    int T,
    int N,
    int _L,
    const Float* _input,
    const int* _target,
    const int* targetSize,
    int* bestPaths,
    void* workspace) {
  const int _S = (2 * _L) + 1;
  const int blank_label = N - 1;
  WorkspacePtrs<Float> ws(workspace, B, T, N, _L);
  for (auto b = 0; b < B; b++) {
    auto L = targetSize[b];
    auto S = (2 * L) + 1;
    int repeats = setup_labels(
        _target + (b * _L),
        ws.s_inc + (b * _S),
        ws.e_inc + (b * _S),
        ws.labels_w_blanks + (b * _S),
        blank_label,
        L,
        S);
    compute_alphas(
        _input + (b * N * T),
        repeats,
        S,
        T,
        N,
        ws.e_inc + b * _S,
        ws.s_inc + b * _S,
        ws.labels_w_blanks + b * _S,
        ws.alpha + (b * _S * T),
        ws.backptr + (b * _S * T),
        bestPaths + (b * T));
  }
}

template struct ConnectionistTemporalClassificationCriterion<float>;
template struct ConnectionistTemporalClassificationCriterion<double>;

} // namespace cpu
} // namespace w2l
