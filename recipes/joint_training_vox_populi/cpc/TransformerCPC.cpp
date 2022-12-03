/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "TransformerCPC.h"

#include <arrayfire.h>

#include "flashlight/fl/autograd/Functions.h"
#include "flashlight/fl/contrib/modules/Transformer.h"
#include "flashlight/fl/nn/Init.h"
#include "flashlight/fl/nn/Utils.h"

#include <cmath>

namespace {
fl::Variable
transformerInitLinear(int32_t inDim, int32_t outDim, float gain = 1.0) {
  // float std = std::sqrt(1.0 / float(inDim));
  float std = gain * std::sqrt(6.0 / (float(inDim) + float(outDim)));
  return fl::uniform(outDim, inDim, -std, std, af::dtype::f32, true);
}

fl::Variable
transformerInitLinearBias(int32_t inDim, int32_t outDim, bool zero = false) {
  float std = std::sqrt(1.0 / float(inDim));
  if (zero) {
    std = 0;
  }
  return fl::uniform(af::dim4(outDim), -std, std);
}

} // namespace

namespace w2l {
namespace cpc {

TransformerCPC::TransformerCPC(
    int32_t modelDim,
    int32_t headDim,
    int32_t mlpDim,
    int32_t nHeads,
    int32_t bptt,
    float pDropout,
    float pLayerdrop,
    bool useMask,
    bool preLN,
    double layerNormEps)
    : nHeads_(nHeads),
      bptt_(bptt),
      pDropout_(pDropout),
      pLayerdrop_(pLayerdrop),
      useMask_(useMask),
      preLN_(preLN),
      layerNormEps_(layerNormEps),
      w1_(std::make_shared<Linear>(modelDim, mlpDim)),
      w2_(std::make_shared<Linear>(mlpDim, modelDim)),
      wq_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads, 0.707),
          transformerInitLinearBias(modelDim, headDim * nHeads))),
      wk_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads, 0.707),
          transformerInitLinearBias(modelDim, headDim * nHeads))),
      wv_(std::make_shared<Linear>(
          transformerInitLinear(modelDim, headDim * nHeads, 0.707),
          transformerInitLinearBias(modelDim, headDim * nHeads))),
      wf_(std::make_shared<Linear>(
          transformerInitLinear(headDim * nHeads, modelDim),
          transformerInitLinearBias(headDim * nHeads, modelDim, true))),
      norm1_(
          std::make_shared<LayerNorm>(std::vector<int>({0, 3}), layerNormEps_)),
      norm2_(std::make_shared<LayerNorm>(
          std::vector<int>({0, 3}),
          layerNormEps_)) {
  if (bptt > 0) {
    params_.push_back(
        uniform(2 * bptt - 1, headDim, -0.1, 0.1, af::dtype::f32, true));
  }

  add(w1_);
  add(w2_);
  add(wq_);
  add(wk_);
  add(wv_);
  add(wf_);
  add(norm1_);
  add(norm2_);
}

Variable TransformerCPC::mlp(const Variable& input) {
  float pDropout = train_ ? pDropout_ : 0.0;
  // return (*w2_)(dropout(relu((*w1_)(input)), pDropout));
  return (*w2_)(dropout(relu((*w1_)(input)), 0.0));
}

Variable TransformerCPC::getMask(int32_t n, bool cache) {
  auto mask = af::lower(af::constant(1.0, n, n), true);
  if (cache) {
    auto maskCache = af::upper(af::constant(1.0, n, n));
    mask = af::join(1, maskCache, mask);
  }
  return Variable(af::log(mask), false);
}

Variable TransformerCPC::selfAttention(const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  auto encoderInput = input.at(input.size() - 2);
  // in case of previous state input[0] has size CxT_prevxB
  int n = input[0].dims(1), bsz = input[0].dims(2);
  double pDrop = train_ ? pDropout_ : 0.0;

  auto q = transpose((*wq_)(encoderInput));
  std::vector<fl::Variable> inputWithState(input.begin(), input.end() - 1);
  auto k = transpose((*wk_)(concatenate(inputWithState, 1)));
  auto v = transpose((*wv_)(concatenate(inputWithState, 1)));

  q = q / std::sqrt(float(q.dims(1) / nHeads_));

  Variable mask, posEmb;
  if (bptt_ > 0) {
    posEmb =
        tile(params_[0].as(encoderInput.type()), af::dim4(1, 1, nHeads_ * bsz));
  }
  if (useMask_ && encoderInput.dims(1) > 1) {
    // mask future if we use the previous state (then n is previous time)
    mask = getMask(n, input.size() == 3);
  }

  int offset = (input.size() == 2) ? 0 : n;

  // time x batch
  fl::Variable padMask;
  if (!input.back().isempty()) {
    auto padMaskArr = input.back().array();
    padMaskArr =
        af::resize(padMaskArr, encoderInput.dims(1), encoderInput.dims(2));
    padMask = fl::Variable(af::log(padMaskArr), false);
  }
  auto result = multiheadAttention(
      q, k, v, posEmb, mask, padMask, nHeads_, pDrop, offset);
  result = (*wf_)(transpose(result));

  return result;
}

std::vector<Variable> TransformerCPC::forward(
    const std::vector<Variable>& input) {
  // previous step[optionally], input, padMask
  // padMask should be empty if previous step is provided
  // padMask is expected to have "1" on the used positions and "0" on padded
  // positions
  if (input.size() < 2) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: there should be at least input and mask");
  }
  auto x = input.at(input.size() - 2);
  if (!input.back().isempty() && x.dims(2) != input.back().dims(1)) {
    throw std::invalid_argument(
        "Invalid inputs for transformer block: input and Mask batch sizes are different");
  }

  float f = 1.0;
  if (train_ && (af::randu(1).scalar<float>() < pLayerdrop_)) {
    f = 0.0;
  }
  if (preLN_) {
    auto h = (f * (*norm1_)(selfAttention(input))).as(x.type()) + x;
    return {f * (*norm2_)(mlp(h)).as(h.type()) + h};
  } else {
    auto h = (*norm1_)((f * selfAttention(input)).as(x.type()) + x);
    return {(*norm2_)((f * mlp(h)).as(h.type()) + h)};
  }
}

std::string TransformerCPC::prettyString() const {
  std::ostringstream ss;
  ss << "Transformer (nHeads: " << nHeads_ << "), "
     << "(pDropout: " << pDropout_ << "), "
     << "(pLayerdrop: " << pLayerdrop_ << "), "
     << "(bptt: " << bptt_ << "), "
     << "(useMask: " << useMask_ << "), "
     << "(preLayerNorm: " << preLN_ << ")";
  return ss.str();
}

TransformerCPC::TransformerCPC() {}

} // namespace cpc
} // namespace w2l
