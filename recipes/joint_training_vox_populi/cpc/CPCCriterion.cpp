/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "CPCCriterion.h"

#include <arrayfire.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <vector>
#include "flashlight/pkg/speech/criterion/CriterionUtils.h"

using namespace fl;

namespace w2l {

void PartialLoading(
    int n_layers,
    std::shared_ptr<fl::Sequential> net0,
    std::shared_ptr<fl::Sequential> net) {
  auto modules_0 = net0->modules();

  if (n_layers < 0) {
    n_layers = modules_0.size() + n_layers;
  }

  for (int i = 0; i < n_layers; i++) {
    net->add(modules_0[i]);
  }
}

CPCCriterion::CPCCriterion(
    int nEncoder,
    int nContext,
    int nMutual,
    int nOffset,
    int nUnits,
    int nPieces,
    int nNegative,
    int nBuffer,
    float temperature)
    : nEncoder_(nEncoder),
      nContext_(nContext),
      nMutual_(nMutual),
      nOffset_(nOffset),
      nUnits_(nUnits),
      nPieces_(nPieces),
      nNegative_(nNegative),
      nBuffer_(nBuffer),
      temperature_(temperature) {
  params_.push_back(uniform(nEncoder_, 1, -1, 1));
  // linear layers for computing mutual information between
  // encoder and context features
  add(std::make_shared<Linear>(nEncoder_, nMutual_));
  add(std::make_shared<Linear>(nContext_, nMutual_));
}

af::array CPCCriterion::getRandomIntegers(int N) {
  auto rnd = af::randu(N) * 100000;
  return rnd.as(s64);
}

af::array shift_non_circular(const af::array& inp, int delta) {
  int T = inp.dims(1);
  int abs_delta = delta;
  if (delta < 0) {
    abs_delta = -delta;
  }
  auto pad_inp = af::pad(
      inp,
      af::dim4(0, abs_delta, 0, 0),
      af::dim4(0, abs_delta, 0, 0),
      AF_PAD_ZERO);
  auto out = af::shift(pad_inp, 0, delta);
  out = out(af::span, af::seq(abs_delta, T + abs_delta - 1));
  return out;
}

// apply mask on input using supplied masking parameters
// also internally store masking indices
Variable
CPCCriterion::getMask(const Variable& input, float mask_prob, int mask_length) {
  int C = input.dims(0);
  int T = input.dims(1);
  int N = input.dims(2);
  af::dim4 maskDims = af::dim4(1, T, N);

  af::array randMatrix;
  af::array midMask, sumMask;
  Variable totalMask;

  int numMask = (mask_prob * T);
  midMask = af::constant(0., af::dim4(1, T, N), f32);
  for (int i = 0; i < N; i++) {
    auto startIdx = getRandomIntegers(numMask) % T;
    midMask(af::span, startIdx, i) = 1.;
  }

  sumMask = midMask * 1.;
  int delta;
  int pow = 1;
  for (int i = 1; i < mask_length; i++) {
    delta = pow * (i + 1) / 2;
    sumMask = sumMask + shift_non_circular(midMask, delta);
    pow *= -1;
  }
  sumMask = af::min(sumMask, 1.);
  // masked_ = Variable(midMask*1., false);
  auto mask = af::moddims(sumMask, af::dim4(T, N));

  // restrict masking by min len across batches
  int minLen = af::min<int>(af::sum(mask, 0));
  auto maskMinLen = af::constant(0., af::dim4(T, N), f32);
  for (int i = 0; i < N; i++) {
    auto maskIdx = af::where(mask(af::span, i));
    auto tmp = af::randu(maskIdx.dims(0));
    af::array val, idx;
    af::sort(val, idx, tmp);
    idx = idx(af::seq(0, minLen - 1));
    maskIdx = maskIdx(idx);
    maskMinLen(maskIdx, i) = 1.;
  }
  mask = af::moddims(maskMinLen, af::dim4(1, T, N));

  masked_ = Variable(mask, false);
  totalMask = tileAs(Variable(mask, false), input.dims());
  auto maskEmbedding = tileAs(params_[0], input.dims());

  auto inputMasked = input * (1 - totalMask) + maskEmbedding * totalMask;
  return inputMasked;
}

Variable CPCCriterion::getNegativeSamples(const Variable& inp) {
  int C = inp.dims(0);
  int T = inp.dims(1);
  int N = inp.dims(2);

  int nNeg = T;
  // int nBuff = nBuffer_;
  int nBuff = 1;
  if (nNeg > nNegative_)
    nNeg = nNegative_;

  // exclude current position with window
  auto time_idx = af::range(af::dim4(T, N, nNeg), 0, s64);
  auto min_idx = af::min(T, 1 + nBuff + time_idx);
  auto max_idx = af::max(T, T - nBuff + time_idx);
  auto mod_idx = max_idx - min_idx;
  auto rnd_idx =
      af::moddims(getRandomIntegers(T * N * nNeg), af::dim4(T, N, nNeg)) %
      mod_idx;
  time_idx = (min_idx + rnd_idx) % T;

  // current sequence only
  auto batch_idx = af::range(af::dim4(T, N, nNeg), 1, s64);
  batch_idx = batch_idx % N;

  auto idx = af::flat(time_idx + batch_idx * T);
  auto out = moddims(inp, af::dim4(C, T * N));
  out = out(af::span, idx);

  out = moddims(out, af::dim4(C, T, N, nNeg));

  return out;
}

/* C = number of channels
 * T = number of time frames
 * N = number of elements in batch
 */

std::vector<Variable> CPCCriterion::forward(
    const std::vector<Variable>& inputs) {
  // enc_out, context = C T N 1
  const auto& samples = inputs[0];
  const auto& context_mask = inputs[1];

  int N = context_mask.dims(2);
  int T = context_mask.dims(1);
  int C = context_mask.dims(0);

  std::vector<Variable> loss;

  auto mask = masked_.array();

  for (int i = 0; i < N; i++) {
    auto batch_mask = af::where(af::flat(mask(af::span, af::span, i))).as(s64);
    auto anchor =
        mutualLinear(1)
            ->forward(context_mask(af::span, batch_mask, i, af::span, true))
            .as(f32);
    auto pos_samples =
        mutualLinear(0)
            ->forward(samples(af::span, batch_mask, i, af::span, true))
            .as(f32);

    anchor = anchor / tileAs(norm(anchor, {0}), anchor.dims());
    pos_samples =
        pos_samples / tileAs(norm(pos_samples, {0}), pos_samples.dims());

    auto neg_samples = getNegativeSamples(pos_samples);
    auto anchor_neg = tileAs(anchor, neg_samples.dims());

    pos_samples = sum(pos_samples * anchor, {0}) / temperature_;
    neg_samples = sum(neg_samples * anchor_neg, {0}) / temperature_;
    auto all_samples = concatenate({pos_samples, neg_samples}, 3);

    auto max_samples = Variable(af::max(all_samples.array(), 3), false);
    auto sum_samples =
        sum(exp(all_samples - tileAs(max_samples, all_samples)), {3});
    loss.push_back(
        sum((max_samples + log(sum_samples) - pos_samples), {1}) /
        anchor.dims(1));
  }

  return {reorder(concatenate(loss, 2), 2, 0, 1, 3)};
}

af::array CPCCriterion::viterbiPath(
    const af::array& input,
    const af::array& inputSize) {
  std::cout << "Should not be here" << std::endl;
  exit(1);
  return input * 0;
} // namespace w2l

std::string CPCCriterion::prettyString() const {
  return "CPCCriterion";
}

} // namespace w2l
