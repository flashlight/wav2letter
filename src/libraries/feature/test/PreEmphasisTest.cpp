/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/PreEmphasis.h"

using w2l::PreEmphasis;

// Matlab code used: B=[1, -0.95]; = filter(B, 1, data, [], 2);
// For first element in data multiply by (1 - alpha)
TEST(PreEmphasisTest, matlabCompareTest) {
  int N = 8;
  PreEmphasis preemphasis1d(0.95, N);
  std::vector<float> input{0.098589,
                           0.715877,
                           0.750572,
                           0.787636,
                           0.116829,
                           0.242914,
                           0.327526,
                           0.410389};
  // numdims = 1
  std::vector<float> matlaboutput1d{0.004929,
                                    0.622218,
                                    0.070489,
                                    0.074592,
                                    -0.631425,
                                    0.131927,
                                    0.096757,
                                    0.099240};
  auto output1d = preemphasis1d.apply(input);
  // Implementation should match with matlab.
  ASSERT_TRUE(compareVec<float>(output1d, matlaboutput1d));

  // numdims = 2
  PreEmphasis preemphasis2d(0.95, N / 2);
  std::vector<float> matlaboutput2d{0.004929,
                                    0.622218,
                                    0.070489,
                                    0.074592,
                                    0.005841,
                                    0.131927,
                                    0.096757,
                                    0.099240};
  auto output2d = preemphasis2d.apply(input);
  // Implementation should match with matlab.
  ASSERT_TRUE(compareVec<float>(output2d, matlaboutput2d));
}

TEST(PreEmphasisTest, batchingTest) {
  int N = 16, B = 15;
  auto input = randVec<float>(N * B);
  auto preemphasis = PreEmphasis(0.5, N);
  auto output = preemphasis.apply(input);
  ASSERT_EQ(output.size(), input.size());
  for (int i = 0; i < B; ++i) {
    std::vector<float> curInput(N), expOutput(N);
    std::copy(
        input.data() + i * N, input.data() + (i + 1) * N, curInput.data());
    std::copy(
        output.data() + i * N, output.data() + (i + 1) * N, expOutput.data());
    auto curOutput = preemphasis.apply(curInput);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-10));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
