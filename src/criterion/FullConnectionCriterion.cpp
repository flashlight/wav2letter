/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "FullConnectionCriterion.h"

#include "CriterionUtils.h"
#include "math.h"

namespace fl {

FullConnectionCriterion::FullConnectionCriterion(
    intl N,
    w2l::CriterionScaleMode scalemode)
    : N_(N), scaleMode_(scalemode) {
  if (N_ <= 0) {
    throw(af::exception("FCC: Size of transition matrix is less than 0."));
  }
  auto transition = constant(0.0, af::dim4(N_, N_));
  params_ = {transition};
}

Variable FullConnectionCriterion::forward(
    const Variable& input,
    const Variable& target) {
  const intl N = input.dims(0);
  const intl T = input.dims(1);
  const intl B = input.dims(2);
  const intl L = target.dims(0);
  if (N != N_) {
    throw(af::exception("FCC: N doesn't match with the letter size."));
  }

  /* Forward */
  auto fwBuf = fwParams(N, T, B, L);
  target.host(fwBuf.targetsRaw.data());
  input.host(fwBuf.inputsRaw.data());
  params_[0].host(fwBuf.transRaw.data());

  auto scaleFn = getCriterionScaleFn(scaleMode_);

#pragma omp parallel for num_threads(B)
  for (intl b = 0; b < B; b++) {
    auto targets = fwBuf.targetsRaw.data() + b * L;
    intl TN = w2l::getTargetSize(targets, L);
    TN = std::min(TN, T);
    if (TN == 0) {
      throw(af::exception("Target size cannot be empty for FCC"));
    }
    fwBuf.scale[b] = scaleFn(N, T, TN);
    double* alpha = fwBuf.alpha.data() + b * N * T;
    intl* alphaIndex = fwBuf.alphaIndex.data() + b * N * T;
    float* inputsRaw = fwBuf.inputsRaw.data() + b * N * T;

    for (intl i = 0; i < N; i++) {
      alpha[i] = inputsRaw[i];
    }

    double* alphaCurFrame;
    intl* alphaIndexCurFrame;
    for (intl t = 1; t < T; t++) {
      double* alphaPrevFrame = alpha + (t - 1) * N;
      alphaCurFrame = alpha + t * N;
      alphaIndexCurFrame = alphaIndex + t * N;
      const float* inputs = inputsRaw + t * N;
      for (intl i = 0; i < N; i++) {
        double sum = 0, max = NEG_INFINITY_DBL;
        for (intl j = 0; j < N; j++) {
          double z = fwBuf.transRaw[i * N + j] + alphaPrevFrame[j];
          if (max < z) {
            alphaIndexCurFrame[i] = j;
            max = z;
          }
        }
        for (intl j = 0; j < N; j++) {
          double z = fwBuf.transRaw[i * N + j] + alphaPrevFrame[j];
          sum += std::exp(z - max);
        }

        alphaCurFrame[i] = max + std::log(sum) + inputs[i];
      }
    }

    alphaCurFrame = alpha + (T - 1) * N;
    double sum = 0, max = NEG_INFINITY_DBL;
    for (long i = 0; i < N; i++) {
      if (max < alphaCurFrame[i]) {
        max = alphaCurFrame[i];
      }
    }
    for (intl i = 0; i < N; i++) {
      sum += std::exp(alphaCurFrame[i] - max);
    }
    fwBuf.res[b] = static_cast<float>((std::log(sum) + max) * fwBuf.scale[b]);
  }

  auto result = af::array(B, fwBuf.res.data());

  /* Backward */
  auto gradFunc = [B, N, T, fwBuf](
                      std::vector<Variable>& inputs,
                      const Variable& gradOutput) {
    auto bwBuf = bwParams(N, T, B);
    gradOutput.host(bwBuf.outputsGrad.data());

#pragma omp parallel for num_threads(B)
    for (intl b = 0; b < B; b++) {
      const float grad = fwBuf.scale[b] * bwBuf.outputsGrad[b];
      float* inputsGrad = bwBuf.inputsGrad.data() + b * N * T;
      double* alphaGrad = bwBuf.alphaGrad.data() + b * N * T;
      const double* alpha = fwBuf.alpha.data() + b * N * T;
      const intl* alphaIndex = fwBuf.alphaIndex.data() + b * N * T;
      double* transGrad = bwBuf.transGrad.data() + b * N * N;

      // bw step 1
      {
        const double* alphaCurFrame = alpha + (T - 1) * N;
        double* alphaGradCurFrame = alphaGrad + (T - 1) * N;
        float* inputsGradCurFrame = inputsGrad + (T - 1) * N;
        double max = NEG_INFINITY_DBL;
        for (intl j = 0; j < N; j++) {
          if (max < alphaCurFrame[j]) {
            max = alphaCurFrame[j];
          }
        }

        double alphaGradSum = 0;
        for (intl j = 0; j < N; j++) {
          alphaGradSum += std::exp(alphaCurFrame[j] - max);
        }
        for (intl j = 0; j < N; j++) {
          alphaGradCurFrame[j] =
              std::exp(alphaCurFrame[j] - max) / alphaGradSum;
          inputsGradCurFrame[j] = alphaGradCurFrame[j] * grad;
        }
      }

      // bw
      for (intl t = T - 2; t >= 0; t--) {
        const double* alphaCurFrame = alpha + t * N;
        double* alphaGradCurFrame = alphaGrad + t * N;
        double* alphaGradPrevFrame = alphaGrad + (t + 1) * N;
        const intl* alphaIndexCurFrame = alphaIndex + (t + 1) * N;
        float* inputsGradCurFrame = inputsGrad + t * N;

        std::vector<double> m(N * N);
        for (intl i = 0; i < N; i++) {
          double max = fwBuf.transRaw[i * N + alphaIndexCurFrame[i]] +
              alphaCurFrame[alphaIndexCurFrame[i]];
          double alphaGradSum = 0;
          for (intl j = 0; j < N; j++) {
            m[i * N + j] =
                std::exp(fwBuf.transRaw[i * N + j] + alphaCurFrame[j] - max);
            alphaGradSum += m[i * N + j];
          }
          for (intl j = 0; j < N; j++) {
            m[i * N + j] = m[i * N + j] / alphaGradSum;
          }
        }

        for (intl i = 0; i < N; i++) {
          for (intl j = 0; j < N; j++) {
            alphaGradCurFrame[i] += m[j * N + i] * alphaGradPrevFrame[j];
            transGrad[j * N + i] += m[j * N + i] * alphaGradPrevFrame[j] * grad;
          }
          inputsGradCurFrame[i] = alphaGradCurFrame[i] * grad;
        }
      }
    }

    for (intl b = 0; b < B; b++) {
      double* transGrad = bwBuf.transGrad.data() + b * N * N;
      for (intl i = 0; i < N * N; i++) {
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

std::string FullConnectionCriterion::prettyString() const {
  return "FullConnectionCriterion";
}

} // namespace fl
