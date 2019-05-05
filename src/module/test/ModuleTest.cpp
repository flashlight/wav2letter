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

#include "common/Utils.h"
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
