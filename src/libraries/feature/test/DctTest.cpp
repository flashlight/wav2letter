/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/Dct.h"

using w2l::Dct;

// Matlab code used:
// dctm = @( N, M )( sqrt(2.0/M) * cos( repmat([0:N-1].',1,M) ...
//                                  .* repmat(pi*([1:M]-0.5)/M,N,1) ) );
// DCT * IN;
// Reference: Kamil Wojcicki, HTK MFCC MATLAB, URL:
//    https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
TEST(DctTest, matlabCompareTest) {
  // Test Case: 1
  Dct dct1(9, 6);
  std::vector<float> input1(9, 1.0);
  std::vector<float> matlaboutput1{4.24264, 0.0, 0.0, 0.0, 0.0, 0.0};
  auto output1 = dct1.apply(input1);
  // Implementation should match with matlab for Test case 1.
  ASSERT_TRUE(compareVec(output1, matlaboutput1));

  // Test Case: 2
  Dct dct2(40, 23);
  std::vector<float> input2{
      3.827583, 3.975999, 0.9343630, 2.448821, 2.227931,  3.231565,  3.546824,
      3.773433, 1.380125, 3.398513,  3.275490, 0.8130586, 0.5949884, 2.491820,
      4.798719, 1.701928, 2.926338,  1.119059, 3.756335,  1.275475,  2.529785,
      3.495383, 4.454516, 4.796457,  2.736077, 0.6931222, 0.7464700, 1.287541,
      4.203586, 1.271410, 4.071424,  1.217624, 4.646318,  1.749918,  0.9829762,
      1.255419, 3.080223, 2.366444,  1.758297, 4.154143};
  std::vector<float> matlaboutput2{
      23.03049,    0.7171224,  0.09039740, 0.5560513, 1.210070,  -0.6701894,
      -0.7615307,  0.1116579,  1.157483,   -2.012746, 2.964205,  2.444191,
      -0.4926429,  -0.1332636, 1.275104,   0.2767147, 0.2781188, 2.661390,
      -0.03644234, -2.326455,  -0.1963445, -1.229159, 2.124846};
  auto output2 = dct2.apply(input2);
  // Implementation should match with matlab for Test case 2.
  ASSERT_TRUE(compareVec(output2, matlaboutput2));
}

TEST(DctTest, batchingTest) {
  int F = 16, C = 10, B = 15;
  auto input = randVec<float>(F * B);
  auto dct = Dct(F, C);
  auto output = dct.apply(input);
  ASSERT_EQ(output.size(), C * B);
  for (int i = 0; i < B; ++i) {
    std::vector<float> curInput(F), expOutput(C);
    std::copy(
        input.data() + i * F, input.data() + (i + 1) * F, curInput.data());
    std::copy(
        output.data() + i * C, output.data() + (i + 1) * C, expOutput.data());
    auto curOutput = dct.apply(curInput);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-5));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
