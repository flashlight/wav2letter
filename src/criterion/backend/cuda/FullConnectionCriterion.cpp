#include "criterion/FullConnectionCriterion.h"

#include <cassert>
#include <cmath>

#include <flashlight/common/cuda.h>

#include "criterion/CriterionUtils.h"
#include "criterion/backend/cuda/kernels/FullConnectionCriterion.cuh"

using namespace af;

namespace w2l {

static void backward(
    std::vector<fl::Variable>& inputs,
    const fl::Variable& grad_output,
    int BB,
    int B,
    int N,
    int T,
    const array& filter_idxs,
    const array& fccacc,
    const array& scale) {
  assert(inputs.size() == 2);
  const auto& gscale = scale * grad_output.array()(filter_idxs); // [B]
  const auto& trans = inputs[1].array(); // [N, N]
  array fccgacc(N, B, T, f64);
  auto gtrans = constant(0, N, N, B, f64);

  const auto& final_em = fccacc(span, span, T - 1); // [N, B]
  const auto& final_max = max(final_em, 0); // [1, B]
  const auto& final_exp = exp(final_em - tile(final_max, N)); // [N, B]
  const auto& final_dlse = final_exp / tile(sum(final_exp, 0), N); // [N, B]
  fccgacc(span, span, T - 1) = final_dlse;

  {
    fl::DevicePtr trans_raw(trans);
    fl::DevicePtr fccacc_raw(fccacc);
    fl::DevicePtr fccgacc_raw(fccgacc);
    fl::DevicePtr gtrans_raw(gtrans);
    if (w2l::cuda::fullConnectionCriterionBackward(
            T,
            B,
            N,
            static_cast<const float*>(trans_raw.get()),
            static_cast<const double*>(fccacc_raw.get()),
            static_cast<double*>(fccgacc_raw.get()),
            static_cast<double*>(gtrans_raw.get()),
            fl::cuda::getActiveStream())) {
      throw std::runtime_error("FCC: Backward pass failed");
    }
  }

  const auto& gem = fccgacc * tile(moddims(gscale, 1, B), N, 1, T); // [N, B, T]
  auto gem_r = constant(0, N, T, BB, f32);
  gem_r(span, span, filter_idxs) = w2l::reorder(gem, 0, 2, 1).as(f32);
  auto gtrans_r = sum(gtrans * tile(moddims(gscale, 1, 1, B), N, N), 2).as(f32);

  inputs[0].addGrad(fl::Variable(gem_r, false));
  inputs[1].addGrad(fl::Variable(gtrans_r, false));
}

fl::Variable FullConnectionCriterion::forward(
    const fl::Variable& input,
    const fl::Variable& target) {
  int N = input.dims(0);
  int T = input.dims(1);
  int BB = input.dims(2);
  int L = target.dims(0);

  const auto& transitions = param(0);
  if (N != transitions.dims(0)) {
    throw std::runtime_error("FCC: Transition dims don't match N.");
  }

  auto scaleFn = getCriterionScaleFn(scaleMode_);

  std::vector<int> target_host(BB * L);
  std::vector<double> scale_host;
  std::vector<int> filter_idxs_host;

  target.host(target_host.data());
  for (int b = 0; b < BB; b++) {
    const auto target_p = target_host.data() + b * L;
    int TN = getTargetSize(target_p, L);
    if (TN > T || TN == 0) {
      continue;
    }
    filter_idxs_host.push_back(b);
    scale_host.push_back(scaleFn(N, T, TN));
  }

  int B = filter_idxs_host.size();
  if (B == 0) {
    auto grad_func =
        [BB, N, T](std::vector<fl::Variable>& inputs, const fl::Variable&) {
          inputs[0].addGrad(fl::constant(0, dim4(N, T, BB), f32, false));
          inputs[1].addGrad(fl::constant(0, dim4(N, N), f32, false));
        };
    return fl::Variable(
        constant(NAN, BB, f32), {input, transitions}, grad_func);
  }

  array filter_idxs(B, filter_idxs_host.data());
  array scale(B, scale_host.data());
  array inp(w2l::reorder(input.array(), 0, 2, 1)(
      span, filter_idxs, span)); // [N, B, T]
  array fccacc(N, B, T, f64);
  fccacc(span, span, 0) = inp(span, span, 0);

  {
    fl::DevicePtr inp_raw(inp);
    fl::DevicePtr trans_raw(transitions.array());
    fl::DevicePtr fccacc_raw(fccacc);
    if (w2l::cuda::fullConnectionCriterionForward(
            T,
            B,
            N,
            static_cast<const float*>(inp_raw.get()),
            static_cast<const float*>(trans_raw.get()),
            static_cast<double*>(fccacc_raw.get()),
            fl::cuda::getActiveStream())) {
      throw std::runtime_error("FCC: Forward pass failed");
    }
  }

  const auto& final_em = fccacc(span, span, T - 1); // [N, B]
  const auto& final_max = max(final_em, 0); // [1, B]
  const auto& final_lse =
      final_max + log(sum(exp(final_em - tile(final_max, N)), 0)); // [1, B]

  const auto& fcc = moddims(final_lse, B) * scale;
  if (anyTrue<bool>(isNaN(fcc))) {
    throw std::runtime_error("Loss is NaN value");
  }
  auto fcc_r = constant(NAN, BB, f32);
  fcc_r(filter_idxs) = fcc.as(f32);

  auto grad_func = [BB, B, N, T, filter_idxs, fccacc, scale](
                       std::vector<fl::Variable>& inputs,
                       const fl::Variable& grad_output) {
    backward(inputs, grad_output, BB, B, N, T, filter_idxs, fccacc, scale);
  };
  return fl::Variable(fcc_r, {input, transitions}, grad_func);
}

} // namespace w2l
