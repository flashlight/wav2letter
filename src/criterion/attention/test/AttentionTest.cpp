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

namespace {
void sequential_test(std::shared_ptr<AttentionBase> attention, int H) {
  int B = 2, T = 10;

  Variable encodedx(af::randn(H, T, B), true);
  Variable encodedy(af::randn(H, 1, B), true);

  Variable alphas, summaries;
  for (int step = 0; step < 3; ++step) {
    std::tie(alphas, summaries) =
        attention->forward(encodedy, encodedx, alphas);
    ASSERT_EQ(alphas.dims(), af::dim4(1, T, B));
    ASSERT_EQ(summaries.dims(), af::dim4(H, 1, B));

    auto alphasum = sum(alphas.array(), 1);
    auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
    ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
  }

  Variable window_mask = Variable(af::constant(1.0, 1, T, B), false);
  auto alphas1 =
      std::get<0>(attention->forward(encodedy, encodedx, alphas, window_mask));
  auto alphas2 = std::get<0>(attention->forward(encodedy, encodedx, alphas));
  ASSERT_TRUE(allClose(alphas1, alphas2, 1e-6));

  Variable encodedy_invalid(af::randn(H, 10, B), true);
  EXPECT_THROW(
      attention->forward(encodedy_invalid, encodedx, alphas),
      std::invalid_argument);
}

} // namespace

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

TEST(AttentionTest, SimpleLocationAttention) {
  int H = 8, K = 5;
  sequential_test(std::make_shared<SimpleLocationAttention>(K), H);
}

TEST(AttentionTest, LocationAttention) {
  int H = 8, K = 5;
  sequential_test(std::make_shared<LocationAttention>(H, K), H);
}

TEST(AttentionTest, NeuralLocationAttention) {
  int H = 8, A = 8, C = 5, K = 3;
  sequential_test(std::make_shared<NeuralLocationAttention>(H, A, C, K), H);
}

TEST(AttentionTest, MultiHeadContentAttention) {
  int H = 512, B = 2, T = 10, U = 5, NH = 8;

  for (bool keyValue : {true, false}) {
    for (bool splitInput : {true, false}) {
      MultiHeadContentAttention attention(H, NH, keyValue, splitInput);

      auto Hencode = keyValue ? H * 2 : H;
      Variable encodedx(af::randn(Hencode, T, B), true);
      Variable encodedy(af::randn(H, U, B), true);

      Variable alphas, summaries;
      std::tie(alphas, summaries) = attention(encodedy, encodedx, Variable{});
      ASSERT_EQ(alphas.dims(), af::dim4(U * NH, T, B));
      ASSERT_EQ(summaries.dims(), af::dim4(H, U, B));

      auto alphasum = sum(alphas.array(), 1);
      auto ones = af::constant(1.0, alphasum.dims(), alphasum.type());
      ASSERT_TRUE(allClose(alphasum, ones, 1e-5));
    }
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
