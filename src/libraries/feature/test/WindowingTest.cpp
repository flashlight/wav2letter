/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/Windowing.h"

using w2l::Windowing;
using w2l::WindowType;

TEST(WindowingTest, hammingCoeffsTest) {
  int N = 64;
  auto hammwindow = Windowing(N, WindowType::HAMMING);
  std::vector<float> matlabcoeffs{
      0.080000, 0.082286, 0.089121, 0.100437, 0.116121, 0.136018, 0.159930,
      0.187620, 0.218811, 0.253195, 0.290429, 0.330143, 0.371943, 0.415413,
      0.460122, 0.505624, 0.551468, 0.597198, 0.642360, 0.686504, 0.729192,
      0.770000, 0.808522, 0.844375, 0.877204, 0.906681, 0.932514, 0.954446,
      0.972259, 0.985776, 0.994862, 0.999428, 0.999428, 0.994862, 0.985776,
      0.972259, 0.954446, 0.932514, 0.906681, 0.877204, 0.844375, 0.808522,
      0.770000, 0.729192, 0.686504, 0.642360, 0.597198, 0.551468, 0.505624,
      0.460122, 0.415413, 0.371943, 0.330143, 0.290429, 0.253195, 0.218811,
      0.187620, 0.159930, 0.136018, 0.116121, 0.100437, 0.089121, 0.082286,
      0.080000,
  };
  std::vector<float> input(N, 1.0);
  auto output = hammwindow.apply(input);
  // Hamming window coefficients should match with matlab implementation.
  ASSERT_TRUE(compareVec<float>(output, matlabcoeffs));
}

TEST(WindowingTest, hanningCoeffsTest) {
  int N = 32;
  auto hannwindow = Windowing(N, WindowType::HANNING);
  std::vector<float> matlabcoeffs{
      0.00000, 0.01024, 0.04052, 0.08962, 0.15552, 0.23552, 0.32635, 0.42429,
      0.52532, 0.62533, 0.72020, 0.80605, 0.87938, 0.93717, 0.97707, 0.99743,
      0.99743, 0.97707, 0.93717, 0.87938, 0.80605, 0.72020, 0.62533, 0.52532,
      0.42429, 0.32635, 0.23552, 0.15552, 0.08962, 0.04052, 0.01024, 0.00000};
  std::vector<float> input(N, 1.0);
  auto output = hannwindow.apply(input);
  // Hamming window coefficients should match with matlab implementation.
  ASSERT_TRUE(compareVec<float>(output, matlabcoeffs));
}

TEST(WindowingTest, batchingTest) {
  int N = 16, B = 15;
  auto input = randVec<float>(N * B);
  auto hannwindow = Windowing(N, WindowType::HANNING);
  auto output = hannwindow.apply(input);
  ASSERT_EQ(output.size(), input.size());
  for (int i = 0; i < B; ++i) {
    std::vector<float> curInput(N), expOutput(N);
    std::copy(
        input.data() + i * N, input.data() + (i + 1) * N, curInput.data());
    std::copy(
        output.data() + i * N, output.data() + (i + 1) * N, expOutput.data());
    auto curOutput = hannwindow.apply(curInput);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-10));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
