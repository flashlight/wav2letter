/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "ForceAlignmentCriterion.h"

#include "CriterionUtils.h"

using namespace fl;

namespace w2l {

ForceAlignmentCriterion::ForceAlignmentCriterion(
    int N,
    w2l::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw std::invalid_argument(
        "FAC: Size of transition matrix is less than 0.");
  }
  auto transition = constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

Variable ForceAlignmentCriterion::forward(
    const Variable& input,
    const Variable& target) {
  int N = input.dims(0);
  int T = input.dims(1);
  int B = input.dims(2);
  int L = target.dims(0);
  if (N != N_) {
    throw std::invalid_argument("FAC: N doesn't match with the letter size.");
  }

  /* Forward */
  auto fwBuf = fwParams(N, T, B, L);
  target.host(fwBuf.targetsRaw.data());
  input.host(fwBuf.inputsRaw.data());
  params_[0].host(fwBuf.transRaw.data());

  auto scaleFn = getCriterionScaleFn(scaleMode_);

#pragma omp parallel for num_threads(B)
  for (int b = 0; b < B; b++) {
    float* inputs = fwBuf.inputsRaw.data() + b * N * T;
    double* alpha = fwBuf.alpha.data() + b * L * T;
    auto targets = fwBuf.targetsRaw.data() + b * L;
    int TN = w2l::getTargetSize(targets, L);
    TN = std::min(TN, T);
    if (TN == 0) {
      throw std::invalid_argument("Target size cannot be empty for FAC");
    }
    fwBuf.scale[b] = scaleFn(N, T, TN);

    alpha[0] = inputs[targets[0]];

    double* transBuf1 = fwBuf.transBuf1.data() + b * L;
    double* transBuf2 = fwBuf.transBuf2.data() + b * L;

    for (int i = 0; i < TN; i++) {
      transBuf1[i] = fwBuf.transRaw[N * (targets[i]) + targets[i]];
      transBuf2[i] =
          i > 0 ? fwBuf.transRaw[N * (targets[i]) + targets[i - 1]] : 0;
    }
    for (int t = 1; t < T; t++) {
      double* alphaPrevFramep = alpha + (t - 1) * TN;
      double* alphaCurFrame = alpha + t * TN;
      const float* inputsCurFrame = inputs + t * N;
      int high = t < TN ? t : TN;
      int low = T - t < TN ? TN - (T - t) : 1;

      if (T - t >= TN) {
        alphaCurFrame[0] =
            transBuf1[0] + alphaPrevFramep[0] + inputsCurFrame[targets[0]];
      }
      for (int i = low; i < high; i++) {
        double s1 = transBuf1[i] + alphaPrevFramep[i];
        double s2 = transBuf2[i] + alphaPrevFramep[i - 1];
        alphaCurFrame[i] = w2l::logSumExp(s1, s2) + inputsCurFrame[targets[i]];
      }
      if (high < TN) {
        alphaCurFrame[high] = transBuf2[high] + alphaPrevFramep[high - 1] +
            inputsCurFrame[targets[high]];
      }
    }

    fwBuf.res[b] = static_cast<float>(alpha[T * TN - 1] * fwBuf.scale[b]);
  }

  auto result = af::array(B, fwBuf.res.data());

  /* Backward */
  auto gradFunc = [B, N, T, L, fwBuf](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto bwBuf = bwParams(N, T, B, L);
    gradOutput.host(bwBuf.outputsGrad.data());

#pragma omp parallel for num_threads(B)
    for (int b = 0; b < B; b++) {
      const float grad = fwBuf.scale[b] * bwBuf.outputsGrad[b];
      float* inputsGrad = bwBuf.inputsGrad.data() + b * N * T;
      double* alphaGrad = bwBuf.alphaGrad.data() + b * L * T;
      double* transGrad = bwBuf.transGrad.data() + b * N * N;
      const double* alpha = fwBuf.alpha.data() + b * L * T;
      auto targets = fwBuf.targetsRaw.data() + b * L;
      int TN = w2l::getTargetSize(targets, L);
      TN = std::min(TN, T);
      if (TN == 0) {
        throw(af::exception("Target size cannot be empty for FAC"));
      }

      double* fwTransBuf1 = bwBuf.fwTransBuf1.data() + b * L;
      double* fwTransBuf2 = bwBuf.fwTransBuf2.data() + b * L;
      double* transBuf1 = bwBuf.transBuf1.data() + b * L;
      double* transBuf2 = bwBuf.transBuf2.data() + b * L;

      for (int i = 0; i < TN; i++) {
        fwTransBuf1[i] = fwBuf.transRaw[N * targets[i] + targets[i]];
        fwTransBuf2[i] =
            i > 0 ? fwBuf.transRaw[N * targets[i] + targets[i - 1]] : 0;
      }
      // bw
      alphaGrad[T * TN - 1] = 1;
      for (int t = T - 1; t > 0; t--) {
        float* inputsCurFrame = inputsGrad + t * N;
        const double* alphaPrevFramep = alpha + (t - 1) * TN;
        double* alphaGradCurFrame = alphaGrad + t * TN;
        double* alphaGradPrevFrame = alphaGrad + (t - 1) * TN;
        int high = t < TN ? t + 1 : TN;
        int low = T - t < TN ? TN - (T - t) : 0;

        for (int i = low; i < high; i++) {
          inputsCurFrame[targets[i]] += grad * alphaGradCurFrame[i];

          if ((high < TN || t == TN - 1) && i == high - 1 && i > 0) {
            alphaGradPrevFrame[i - 1] += alphaGradCurFrame[i];
            transBuf2[i] += alphaGradCurFrame[i];
          } else if (i == 0) {
            alphaGradPrevFrame[i] += alphaGradCurFrame[i];
            transBuf1[i] += alphaGradCurFrame[i];
          } else {
            double m_1 = fwTransBuf1[i] + alphaPrevFramep[i];
            double m_2 = fwTransBuf2[i] + alphaPrevFramep[i - 1];
            double s1 = 0, s2 = 0;
            w2l::dLogSumExp(m_1, m_2, s1, s2, 1);

            transBuf1[i] += s1 * alphaGradCurFrame[i];
            transBuf2[i] += s2 * alphaGradCurFrame[i];

            alphaGradPrevFrame[i] += s1 * alphaGradCurFrame[i];
            alphaGradPrevFrame[i - 1] += s2 * alphaGradCurFrame[i];
          }
        }
      }

      inputsGrad[targets[0]] += alphaGrad[0] * grad;
      for (int i = 0; i < TN; i++) {
        transGrad[(targets[i] * N + targets[i])] += transBuf1[i] * grad;
        if (i > 0) {
          transGrad[(targets[i] * N + targets[i - 1])] += transBuf2[i] * grad;
        }
      }
    }

    for (int b = 0; b < B; b++) {
      double* transGrad = bwBuf.transGrad.data() + b * N * N;
      for (int i = 0; i < N * N; i++) {
        bwBuf.transGradRes[i] += static_cast<float>(transGrad[i]);
      }
    }

    inputs[0].addGrad(
        Variable(af::array(N, T, B, bwBuf.inputsGrad.data()), false));
    inputs[1].addGrad(
        Variable(af::array(N, N, bwBuf.transGradRes.data()), false));
  };

  return Variable(result, {input, params_[0]}, gradFunc);
}

std::string ForceAlignmentCriterion::prettyString() const {
  return "ForceAlignmentCriterion";
}

} // namespace w2l
