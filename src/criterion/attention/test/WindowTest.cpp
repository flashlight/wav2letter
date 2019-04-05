/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <flashlight/flashlight.h>

#include "criterion/attention/window.h"

using namespace fl;
using namespace w2l;

TEST(WindowTest, MedianWindow) {
  int inputsteps = 12;
  int batchsize = 4;
  int hiddendim = 16;
  int w_l = 2;
  int w_r = 3;
  auto input_attn_array = af::abs(af::randn(1, w_l + w_r, batchsize, f32));
  auto input_attn = Variable(
      input_attn_array /
          af::tile(sum(input_attn_array, 1), 1, input_attn_array.dims(1)),
      false);

  MedianWindow window(w_l, w_r);

  // check initialization
  auto mask_0 = window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 0 /* step */);

  auto true_sum_mask_0 = af::constant(0.0, 1, inputsteps, 1, f32);
  true_sum_mask_0(af::span, af::seq(0, w_l + w_r - 1), af::span) = batchsize;

  ASSERT_EQ(mask_0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::sum(mask_0.array(), 2), true_sum_mask_0));

  // check next step
  auto mask_1 =
      window.computeSingleStepWindow(input_attn, inputsteps, batchsize, 1);
  ASSERT_EQ(mask_1.dims(), af::dim4(1, inputsteps, batchsize));

  // make sure large window size is handled
  MedianWindow large_window(100, 100);
  auto mask_large = large_window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 0);
  true_sum_mask_0 = af::constant(batchsize, 1, inputsteps, 1, f32);
  ASSERT_TRUE(allClose(af::sum(mask_large.array(), 2), true_sum_mask_0));
}

TEST(WindowTest, StepWindow) {
  int inputsteps = 100;
  int batchsize = 4;
  int hiddendim = 16;
  int targetlen = 30;
  int s_min = 3, s_max = 15;
  double v_min = 2.3, v_max = 7.5;

  Variable input_attn; // dummy
  std::vector<int> window_boundaries(2, 0);

  StepWindow window(s_min, s_max, v_min, v_max);

  // check initialization
  auto mask_0 = window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 0 /* step */);
  auto true_sum_mask_0 = af::constant(0.0, 1, inputsteps, 1, f32);
  window_boundaries[0] = s_min;
  window_boundaries[1] = s_max;

  true_sum_mask_0(
      af::span,
      af::seq(window_boundaries[0], window_boundaries[1] - 1),
      af::span) = batchsize;

  ASSERT_EQ(mask_0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::sum(mask_0.array(), 2), true_sum_mask_0));

  auto mask_1 = window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 1 /* step */);
  auto true_sum_mask_1 = af::constant(0.0, 1, inputsteps, 1, f32);
  window_boundaries[0] = static_cast<int>(std::round(s_min + v_min));
  window_boundaries[1] = static_cast<int>(std::round(s_max + v_max));

  true_sum_mask_1(
      af::span,
      af::seq(window_boundaries[0], window_boundaries[1] - 1),
      af::span) = batchsize;

  ASSERT_EQ(mask_1.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::sum(mask_1.array(), 2), true_sum_mask_1));

  auto mask_large = window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 1000 /* step */);
  auto true_sum_mask_large = af::constant(0.0, 1, inputsteps, 1, f32);
  window_boundaries[0] = static_cast<int>(std::round(inputsteps - v_max));
  window_boundaries[1] = inputsteps;

  true_sum_mask_large(
      af::span,
      af::seq(window_boundaries[0], window_boundaries[1] - 1),
      af::span) = batchsize;

  ASSERT_EQ(mask_large.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(af::sum(mask_large.array(), 2), true_sum_mask_large));

  auto mask_v = window.computeWindowMask(targetlen, inputsteps, batchsize);
  ASSERT_EQ(mask_v.dims(), af::dim4(targetlen, inputsteps, batchsize));
}

TEST(WindowTest, SoftWindow) {
  int inputsteps = 100;
  int batchsize = 4;
  int targetlen = 15;
  int offset = 10;
  double avg_rate = 5.2, std = 5.0;

  Variable input_attn; // dummy
  SoftWindow window(std, avg_rate, offset);

  auto mask_0 = window.computeSingleStepWindow(
      input_attn, inputsteps, batchsize, 0 /* step */);

  af::array maxv, maxidx;
  max(maxv, maxidx, mask_0.array(), 1);
  std::vector<int> true_maxidx(batchsize, offset);

  ASSERT_EQ(mask_0.dims(), af::dim4(1, inputsteps, batchsize));
  ASSERT_TRUE(allClose(
      maxidx.as(af::dtype::s32),
      af::array(1, 1, batchsize, true_maxidx.data())));

  auto mask_v = window.computeWindowMask(targetlen, inputsteps, batchsize);
  ASSERT_EQ(mask_v.dims(), af::dim4(targetlen, inputsteps, batchsize));
}

TEST(WindowTest, SoftPretrainWindow) {
  int inputsteps = 32;
  int targetlen = 8;
  int batchsize = 4;
  double std = 5.0;

  std::vector<unsigned int> peaks = {0, 4, 8, 12, 16, 20, 24, 28};

  Variable input_attn;
  SoftPretrainWindow window(std);

  // single step
  window.setBatchStat(inputsteps, targetlen, batchsize);
  std::vector<Variable> masks;
  for (int step = 0; step < targetlen; ++step) {
    masks.emplace_back(window.computeSingleStepWindow(
        input_attn, inputsteps, batchsize, step));
  }
  auto mask_s = concatenate(masks, 0);
  af::array maxv, maxid;
  max(maxv, maxid, mask_s.array()(af::span, af::span, 0), 1);

  ASSERT_EQ(mask_s.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(allClose(maxid, af::array(8, peaks.data())));

  // vectorized
  auto mask_v = window.computeWindowMask(targetlen, inputsteps, batchsize);
  max(maxv, maxid, mask_v.array()(af::span, af::span, 0), 1);

  ASSERT_EQ(mask_v.dims(), af::dim4(targetlen, inputsteps, batchsize));
  ASSERT_TRUE(allClose(maxid, af::array(8, peaks.data())));

  ASSERT_TRUE(allClose(mask_s, mask_v));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
