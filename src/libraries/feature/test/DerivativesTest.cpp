/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>
#include <numeric>

#include "TestUtils.h"
#include "libraries/feature/Derivatives.h"

using w2l::Derivatives;

// Reference C++ code taken from HTK - http://htk.eng.cam.ac.uk/
//   float *fp,*fp1,*fp2, *back, *forw;
//   float sum, sigmaT2;
//   int i,t,j;
//
//   sigmaT2 = 0.0;
//   for (t=1;t<=delwin;t++)
//      sigmaT2 += t*t;
//   sigmaT2 *= 2.0;
//   fp = data;
//   for (i=1;i<=n;i++){
//      fp1 = fp; fp2 = fp+offset;
//      for (j=1;j<=vSize;j++){
//         back = forw = fp1; sum = 0.0;
//         for (t=1;t<=delwin;t++) {
//            if (head+i-t > 0)     back -= step;
//            if (tail+n-i+1-t > 0) forw += step;
//            if (!simpleDiffs) sum += t * (*forw - *back);
//         }
//         if (simpleDiffs)
//            *fp2 = (*forw - *back) / (2*delwin);
//         else
//            *fp2 = sum / sigmaT2;
//         ++fp1; ++fp2;
//      }
//      fp += step;
//   }
TEST(DerivativesTest, matlabCompareTest) {
  // Test Case: 1
  Derivatives dev1(4, 4);
  std::vector<float> input1(12);
  std::iota(input1.begin(), input1.end(), 0.0);
  std::vector<float> matlaboutput1{
      0.0000,     1.0000,     2.0000,     3.0000,     4.0000,     5.0000,
      6.0000,     7.0000,     8.0000,     9.0000,     10.0000,    11.0000,
      0.5000000,  0.6666667,  0.8166667,  0.9333333,  1.0000000,  1.0000000,
      1.0000000,  1.0000000,  0.9333333,  0.8166667,  0.6666667,  0.5000000,
      0.0683333,  0.0780556,  0.0794444,  0.0725000,  0.0527778,  0.0180556,
      -0.0180556, -0.0527778, -0.0725000, -0.0794444, -0.0780556, -0.0683333};
  auto output1 = dev1.apply(input1, 1);
  // Implementation should match with matlab for Test case 1.
  ASSERT_TRUE(compareVec<float>(output1, transposeVec(matlaboutput1, 3, 12)));

  // Test Case: 2
  Derivatives dev2(9, 7);
  std::vector<float> input2{
      3.827583, 3.975999, 0.9343630, 2.448821, 2.227931,  3.231565,  3.546824,
      3.773433, 1.380125, 3.398513,  3.275490, 0.8130586, 0.5949884, 2.491820,
      4.798719, 1.701928, 2.926338,  1.119059, 3.756335,  1.275475,  2.529785,
      3.495383, 4.454516, 4.796457,  2.736077, 0.6931222, 0.7464700, 1.287541,
      4.203586, 1.271410, 4.071424,  1.217624, 4.646318,  1.749918,  0.9829762,
      1.255419, 3.080223, 2.366444,  1.758297, 4.154143};
  std::vector<float> matlaboutput2{
      3.827583,   3.975999,   0.9343630,  2.448821,   2.227931,   3.231565,
      3.546824,   3.773433,   1.380125,   3.398513,   3.275490,   0.8130586,
      0.5949884,  2.491820,   4.798719,   1.701928,   2.926338,   1.119059,
      3.756335,   1.275475,   2.529785,   3.495383,   4.454516,   4.796457,
      2.736077,   0.6931222,  0.7464700,  1.287541,   4.203586,   1.271410,
      4.071424,   1.217624,   4.646318,   1.749918,   0.9829762,  1.255419,
      3.080223,   2.366444,   1.758297,   4.154143,   -0.0783472, -0.0703440,
      -0.1002527, -0.1283159, -0.1207580, -0.0744319, -0.0787063, -0.0599186,
      -0.0680858, -0.0298600, -0.0306807, -0.0046153, -0.0141285, 0.0135790,
      0.0392915,  0.0455732,  0.0259977,  0.0162468,  -0.0216384, 0.0220920,
      0.0159542,  0.0143425,  -0.0418714, -0.0117627, 0.0093056,  -0.0307167,
      -0.0436951, -0.0566360, -0.0380197, -0.0700912, -0.0431751, -0.0021685,
      0.0545093,  0.1177130,  0.1458966,  0.1357510,  0.1204694,  0.1087019,
      0.1430639,  0.1260710,  -0.0007462, -0.0001886, 0.0012880,  0.0025709,
      0.0043352,  0.0055983,  0.0073248,  0.0093658,  0.0111437,  0.0122183,
      0.0118505,  0.0093177,  0.0077131,  0.0067685,  0.0053387,  0.0027080,
      0.0005322,  -0.0002259, -0.0021479, -0.0036494, -0.0056067, -0.0061552,
      -0.0065865, -0.0057748, -0.0041803, -0.0013468, 0.0018477,  0.0064985,
      0.0102782,  0.0132019,  0.0138463,  0.0156723,  0.0171224,  0.0170120,
      0.0159708,  0.0139536,  0.0118158,  0.0081756,  0.0046038,  0.0015992};
  auto output2 = dev2.apply(input2, 1);
  // Implementation should match with matlab for Test case 2.
  ASSERT_TRUE(compareVec<float>(output2, transposeVec(matlaboutput2, 3, 40)));
}

TEST(DerivativesTest, batchingTest) {
  int numFeat = 60, frameSz = 20;
  auto input = randVec<float>(numFeat * frameSz);
  Derivatives dev(6, 7);
  auto output = dev.apply(input, numFeat);
  ASSERT_EQ(output.size(), input.size() * 3);
  for (int i = 0; i < numFeat; ++i) {
    std::vector<float> curInput(frameSz), expOutput(frameSz * 3);
    for (int j = 0; j < frameSz; ++j) {
      curInput[j] = input[j * numFeat + i];
      expOutput[j * 3] = output[j * numFeat * 3 + i];
      expOutput[j * 3 + 1] = output[j * numFeat * 3 + numFeat + i];
      expOutput[j * 3 + 2] = output[j * numFeat * 3 + 2 * numFeat + i];
    }
    auto curOutput = dev.apply(curInput, 1);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-4));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
