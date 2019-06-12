/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "criterion/ConnectionistTemporalClassificationCriterion.h"
#include "criterion/CriterionUtils.h"
#include "libraries/criterion/cpu/CriterionUtils.h"

using namespace fl;

using CriterionUtils = w2l::cpu::CriterionUtils<float>;

namespace w2l {

std::vector<Variable> ConnectionistTemporalClassificationCriterion::forward(
    const std::vector<Variable>& inputs) {
  if (inputs.size() != 2) {
    throw std::invalid_argument("Invalid inputs size");
  }
  const auto& input = inputs[0];
  const auto& target = inputs[1];
  validate(input, target);
  auto logprobs = logSoftmax(input, 0);

  std::vector<std::vector<float>> batchAlphas;
  std::vector<float> batchLoss;
  std::vector<float> batchScales;
  std::vector<int> batchTargetSizes;
  {
    const int64_t N = logprobs.dims(0);
    const int64_t T = logprobs.dims(1);
    const int64_t B = logprobs.dims(2);
    const int64_t batchL = target.dims(0);

    batchAlphas.resize(B);
    batchLoss.resize(B);
    batchScales.resize(B);
    batchTargetSizes.resize(B);

    // get host pointers
    std::vector<float> batchInputVec(logprobs.elements());
    logprobs.host(batchInputVec.data());

    std::vector<int> batchTargetVec(target.elements());
    target.host(batchTargetVec.data());

    CriterionUtils::batchTargetSize(
        B, batchL, batchL, batchTargetVec.data(), batchTargetSizes.data());

    CriterionUtils::computeScale(
        B, T, N, scaleMode_, batchTargetSizes.data(), batchScales.data());

#pragma omp parallel for num_threads(B)
    for (int64_t b = 0; b < B; ++b) {
      const float* inputVec = batchInputVec.data() + b * N * T;
      const int* targetVec = batchTargetVec.data() + b * batchL;

      int64_t L = batchTargetSizes[b];
      const int64_t S = 2 * L + 1;
      int64_t R = w2l::countRepeats(targetVec, L);

      // A heuristic to modify target length to be able to compute CTC loss
      L = std::min(L + R, T) - R;
      R = w2l::countRepeats(targetVec, L); // Recompute repeats as L has changed

      auto& alphas = batchAlphas[b];
      alphas.resize(T * S, NEG_INFINITY_FLT);

      int64_t start = (T - (L + R)) > 0 ? 0 : 1;
      int64_t end = (S == 1) ? 1 : 2;

      // base case
      alphas[0] = (start == 0) ? inputVec[N - 1] : NEG_INFINITY_FLT;
      if (S != 1) {
        alphas[1] = inputVec[targetVec[0]];
      }
      for (int64_t t = 1; t < T; ++t) {
        // At each time frame t, only few states can be reached depending
        // on the labels, their ordering and the current time frame.
        if (T - t <= L + R) {
          if (start & 1 && targetVec[start / 2] != targetVec[start / 2 + 1]) {
            ++start;
          }
          ++start;
        }
        if (t <= L + R) {
          if (end % 2 == 0 && end < 2 * L &&
              (targetVec[end / 2 - 1] != targetVec[end / 2])) {
            ++end;
          }
          ++end;
        }
        // Use dynamic programming to recursively compute alphas
        for (int64_t s = start; s < end; ++s) {
          int64_t ts = t * S + s;
          int64_t curLabel = t * N + ((s & 1) ? targetVec[s / 2] : N - 1);
          if (s == 0) {
            alphas[ts] = alphas[ts - S];
          } else if (
              (s % 2 == 0) || s == 1 ||
              targetVec[s / 2] == targetVec[s / 2 - 1]) {
            alphas[ts] = w2l::logSumExp(alphas[ts - S], alphas[ts - S - 1]);
          } else {
            alphas[ts] = w2l::logSumExp(
                alphas[ts - S], alphas[ts - S - 1], alphas[ts - S - 2]);
          }
          alphas[ts] += inputVec[curLabel];
        }
      }
      batchLoss[b] = -w2l::logSumExp(
                         alphas.end()[-1],
                         (S == 1) ? NEG_INFINITY_FLT : alphas.end()[-2]) *
          batchScales[b];
    }
  }
  auto result = af::array(batchLoss.size(), batchLoss.data());

  auto gradFunc = [batchAlphas, batchScales, batchTargetSizes](
                      std::vector<Variable>& moduleInputs,
                      const Variable& gradOutput) {
    const int64_t N = moduleInputs[0].dims(0);
    const int64_t T = moduleInputs[0].dims(1);
    const int64_t B = moduleInputs[0].dims(2);
    const int64_t batchL = moduleInputs[1].dims(0);

    std::vector<float> batchInGrad(moduleInputs[0].elements(), 0.0);

    std::vector<int> batchTargetVec(moduleInputs[1].elements());
    moduleInputs[1].host(batchTargetVec.data());

    std::vector<float> batchOutGrad(gradOutput.elements());
    gradOutput.host(batchOutGrad.data());

#pragma omp parallel for num_threads(B)
    for (int64_t b = 0; b < B; ++b) {
      const int* targetVec = batchTargetVec.data() + b * batchL;
      float* grad = batchInGrad.data() + b * N * T;

      int64_t L = batchTargetSizes[b];

      L = std::min(L, T);
      const int64_t R = w2l::countRepeats(targetVec, L);
      L = std::min(L + R, T) - R;

      const int64_t S = 2 * L + 1;
      const auto& alphas = batchAlphas[b];

      int64_t start = (S == 1) ? S : S - 1;
      int64_t end = S;
      std::vector<float> dAlphas(T * S, 0.0);

      // Compute dAlphas for the last timeframe
      if (S == 1) {
        dAlphas[T * S - 1] = -1.0;
      } else {
        w2l::dLogSumExp(
            alphas[T * S - 2],
            alphas[T * S - 1],
            dAlphas[T * S - 2],
            dAlphas[T * S - 1],
            -1.0);
      }
      float gradScale = batchOutGrad[b] * batchScales[b];

      for (int64_t t = T - 1; t >= 0; --t) {
        // Compute start and end values at time (t) similar to calculation
        // of alpha in CTC forward pass
        if (T - t <= L + R + 1) {
          if (start & 1 && start > 1 &&
              targetVec[start / 2] != targetVec[start / 2 - 1]) {
            --start;
          }
          --start;
        }
        if (t < L + R) {
          if (end % 2 == 0 &&
              (targetVec[end / 2 - 1] != targetVec[end / 2 - 2])) {
            --end;
          }
          --end;
        }
        // Compute grad and dAlphas for (t-1)th frame using chain rule
        for (int64_t s = start; s < end; ++s) {
          int64_t ts = t * S + s;
          int64_t curLabel = t * N + ((s & 1) ? targetVec[s / 2] : N - 1);
          grad[curLabel] += dAlphas[ts] * gradScale;
          if (t == 0) {
            continue;
          }
          if (s == 0) {
            dAlphas[ts - S] += dAlphas[ts];
          } else if (
              (s % 2 == 0) || s == 1 ||
              targetVec[s / 2] == targetVec[s / 2 - 1]) {
            w2l::dLogSumExp(
                alphas[ts - S],
                alphas[ts - S - 1],
                dAlphas[ts - S],
                dAlphas[ts - S - 1],
                dAlphas[ts]);
          } else {
            w2l::dLogSumExp(
                alphas[ts - S],
                alphas[ts - S - 1],
                alphas[ts - S - 2],
                dAlphas[ts - S],
                dAlphas[ts - S - 1],
                dAlphas[ts - S - 2],
                dAlphas[ts]);
          }
        }
      }
    }
    moduleInputs[0].addGrad(
        Variable(af::array(N, T, B, batchInGrad.data()), false));
  };
  return {Variable(result, {logprobs, target}, gradFunc)};
}

} // namespace w2l
