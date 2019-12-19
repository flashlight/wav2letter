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

#include "common/FlashlightUtils.h"
#include "module/module.h"

using namespace fl;
using namespace w2l;

TEST(ModuleTest, TDSFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;

  auto tds = TDSBlock(c, 9, w);
  auto input = Variable(af::randu(timesteps, w, c, batchsize), false);

  auto output = tds.forward({input})[0];

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), w);
  ASSERT_EQ(output.dims(2), c);
}

TEST(ModuleTest, StreamingTDSFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;
  int kw = 9;
  int rPad = 3;

  auto stds =
      TDSBlock(c, kw, w, 0 /* dropout */, 0 /* innerLinearDim */, rPad, true);

  auto input = Variable(af::randu(timesteps, w, c, batchsize), false);

  auto output = stds.forward({input})[0];

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), w);
  ASSERT_EQ(output.dims(2), c);
}

TEST(ModuleTest, SpecAugmentFwd) {
  SpecAugment specAug(0, 27, 2, 100, 0.2, 2);
  int T = 512, F = 80;
  auto input = Variable(af::randu(T, F), false);

  specAug.eval();
  ASSERT_TRUE(fl::allClose(input, specAug(input)));

  specAug.train();
  auto output = specAug(input);
  ASSERT_FALSE(fl::allClose(input, output));

  // Every value of output is either 0 or input
  for (int t = 0; t < T; ++t) {
    for (int f = 0; f < F; ++f) {
      auto o = output.array()(t, f).scalar<float>();
      auto i = input.array()(t, f).scalar<float>();
      ASSERT_TRUE(o == i || o == 0);
    }
  }

  // non-zero time frames are masked
  int tZeros = 0;
  for (int t = 0; t < T; ++t) {
    auto curOutSlice = output.array().row(t);
    tZeros = af::allTrue<bool>(curOutSlice == 0) ? tZeros + 1 : tZeros;
  }
  ASSERT_GT(tZeros, 0);

  // non-zero frequency channels are masked
  int fZeros = 0;
  for (int f = 0; f < F; ++f) {
    auto curOutSlice = output.array().col(f);
    fZeros = af::allTrue<bool>(curOutSlice == 0) ? fZeros + 1 : fZeros;
  }
  ASSERT_GT(fZeros, 0);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
