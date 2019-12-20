/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/Ceplifter.h"

using w2l::Ceplifter;

// matlab code used:
// ceplifter = @( N, L )( 1+0.5*L*sin(pi*[0:N-1]/L) );
// CC = diag( lifter ) * CC;
// Reference: Kamil Wojcicki, HTK MFCC MATLAB, URL:
//    https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
TEST(CeplifterTest, matlabCompareTest) {
  // Test Case: 1
  Ceplifter cep1(25, 22);
  std::vector<float> input1(25, 1.0);
  std::vector<float> matlaboutput1{
      1,        2.565463, 4.099058,   5.569565,  6.947048, 8.203468, 9.313245,
      10.25378, 11.00595, 11.55442,   11.88803,  12,       11.88803, 11.55442,
      11.00595, 10.25378, 9.313245,   8.203468,  6.947048, 5.569565, 4.099058,
      2.565463, 1.000000, -0.5654632, -2.0990581};
  auto output1 = cep1.apply(input1);
  // Implementation should match with matlab for Test case 1.
  ASSERT_TRUE(compareVec<float>(output1, matlaboutput1));
  // Test Case: 2
  Ceplifter cep2(40, 13);
  std::vector<float> input2{
      3.827583, 3.975999, 0.9343630, 2.448821, 2.227931,  3.231565,  3.546824,
      3.773433, 1.380125, 3.398513,  3.275490, 0.8130586, 0.5949884, 2.491820,
      4.798719, 1.701928, 2.926338,  1.119059, 3.756335,  1.275475,  2.529785,
      3.495383, 4.454516, 4.796457,  2.736077, 0.6931222, 0.7464700, 1.287541,
      4.203586, 1.271410, 4.071424,  1.217624, 4.646318,  1.749918,  0.9829762,
      1.255419, 3.080223, 2.366444,  1.758297, 4.154143};
  std::vector<float> matlaboutput2{
      3.82758300,   10.1608714,  3.75679389,  13.0039674,  14.1460142,
      22.8717424,   26.4330877,  28.1219157,  9.76798039,  21.5785018,
      17.3938256,   3.26906521,  1.52052368,  2.49182000,  -2.66593706,
      -3.43908696,  -9.68704871, -4.86722976, -19.0731875, -6.95466478,
      -13.7939251,  -17.7481762, -19.3744501, -15.8776985, -5.52879248,
      -0.385065298, 0.746470000, 3.29037774,  16.9013608,  6.75156506,
      25.8510797,   8.61786241,  34.6271852,  13.0414523,  6.95711783,
      7.97115128,   16.3568998,  9.51476285,  4.49341909,  4.15414300};
  auto output2 = cep2.apply(input2);
  // Implementation should match with matlab for Test case 2.
  ASSERT_TRUE(compareVec<float>(output2, matlaboutput2));
}

TEST(CeplifterTest, batchingTest) {
  int N = 16, B = 15;
  auto input = randVec<float>(N * B);
  auto cep = Ceplifter(N, 25);
  auto output = cep.apply(input);
  ASSERT_EQ(output.size(), input.size());
  for (int i = 0; i < B; ++i) {
    std::vector<float> curInput(N), expOutput(N);
    std::copy(
        input.data() + i * N, input.data() + (i + 1) * N, curInput.data());
    std::copy(
        output.data() + i * N, output.data() + (i + 1) * N, expOutput.data());
    auto curOutput = cep.apply(curInput);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-10));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
