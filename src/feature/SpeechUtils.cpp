/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "SpeechUtils.h"

#include <stdexcept>

#if W2L_LIBRARIES_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace w2l {

template <typename T>
std::vector<T> frameSignal(
    const std::vector<T>& input,
    const FeatureParams& params) {
  auto frameSize = params.numFrameSizeSamples();
  auto frameStride = params.numFrameStrideSamples();
  int64_t numframes = params.numFrames(input.size());
  // HTK: Values coming out of rasta treat samples as integers,
  // not range -1..1, hence scale up here to match (approx)
  T scale = 32768.0;
  std::vector<T> frames(numframes * frameSize);
  for (size_t f = 0; f < numframes; ++f) {
    for (size_t i = 0; i < frameSize; ++i) {
      frames[f * frameSize + i] = scale * input[f * frameStride + i];
    }
  }
  return frames;
}

template <>
std::vector<float> cblasGemm(
    const std::vector<float>& matA,
    const std::vector<float>& matB,
    int n,
    int k) {
  if (n <= 0 || k <= 0 || matA.empty() || (matA.size() % k != 0) ||
      (matB.size() != n * k)) {
    throw std::invalid_argument("cblasGemm: invalid arguments");
  }

  int m = matA.size() / k;

  std::vector<float> matC(m * n);

  cblas_sgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      m,
      n,
      k,
      1.0, // alpha
      matA.data(),
      k,
      matB.data(),
      n,
      0.0, // beta
      matC.data(),
      n);
  return matC;
};

template <>
std::vector<double> cblasGemm(
    const std::vector<double>& matA,
    const std::vector<double>& matB,
    int n,
    int k) {
  if (n <= 0 || k <= 0 || matA.empty() || (matA.size() % k != 0) ||
      (matB.size() != n * k)) {
    throw std::invalid_argument("cblasGemm: invalid arguments");
  }

  int m = matA.size() / k;

  std::vector<double> matC(m * n);

  cblas_dgemm(
      CblasRowMajor,
      CblasNoTrans,
      CblasNoTrans,
      m,
      n,
      k,
      1.0, // alpha
      matA.data(),
      k,
      matB.data(),
      n,
      0.0, // beta
      matC.data(),
      n);

  return matC;
};

template std::vector<float> frameSignal(
    const std::vector<float>&,
    const FeatureParams&);
template std::vector<double> frameSignal(
    const std::vector<double>&,
    const FeatureParams&);
} // namespace w2l
