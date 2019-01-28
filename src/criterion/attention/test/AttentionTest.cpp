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

#include "criterion/attention/attention.h"

using namespace fl;
using namespace w2l;

TEST(AttentionTest, NeuralContentAttention) {
  int H = 8, B = 2, T = 10, U = 5;
  NeuralContentAttention attention(H);

  Variable encodedx(af::randn(H, T, B), true);
  Variable encodedy(af::randn(H, U, B), true);

  Variable alphas, summaries;
  std::tie(alphas, summaries) = attention(encodedy, encodedx, Variable{});
  ASSERT_EQ(alphas.dims(), af::dim4(U, T, B));
  ASSERT_EQ(summaries.dims(), af::dim4(H, U, B));

  auto alphasum = sum(alphas.array(), 1);
  auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
  ASSERT_TRUE(allClose(alphasum, ones, 1e-5));

  Variable window_mask = Variable(af::constant(1.0, U, T, B), false);
  auto alphas1 = std::get<0>(
      attention.forward(encodedy, encodedx, Variable{}, window_mask));
  ASSERT_TRUE(allClose(alphas, alphas1, 1e-6));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
