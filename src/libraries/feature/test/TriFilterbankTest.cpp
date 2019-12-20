/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/TriFilterbank.h"

using w2l::FrequencyScale;
using w2l::TriFilterbank;

// Matlab code used:
// H = trifbank( M, K, R, fs, hz2mel, mel2hz ); % size of H is M x K
// FBE = H * MAG(1:K,:);
// Reference: Kamil Wojcicki, HTK MFCC MATLAB, URL:
//    https://www.mathworks.com/matlabcentral/fileexchange/32849-htk-mfcc-matlab
TEST(TriFilterbankTest, matlabCompareTest) {
  // Test Case: 1
  TriFilterbank triflt1(10, 9, 20000, 0, 10000, FrequencyScale::MEL);
  std::vector<float> matlabfbank1{
      0, 0, 0,        0,        0,        0, 0, 0, 0, 0, 0,
      0, 0, 0.881121, 0.118879, 0,        0, 0, 0, 0, 0, 0,
      0, 0, 0,        0.882891, 0.117109, 0, 0, 0, 0, 0, 0,
      0, 0, 0,        0.569722, 0.430278, 0, 0, 0, 0, 0, 0,
      0, 0, 0,        0.571075, 0.428925, 0, 0, 0, 0, 0, 0,
      0, 0, 0,        0.763933, 0.236067, 0, 0, 0, 0, 0, 0,
      0, 0, 0.082177, 0.917823, 0,        0, 0, 0, 0, 0, 0,
      0, 0, 0.532067, 0,        0,        0, 0, 0, 0, 0, 0,
      0, 0};
  auto outputfbank1 = triflt1.filterbank();
  // Implementation should match with matlab for Test case 1.
  ASSERT_TRUE(compareVec<float>(outputfbank1, matlabfbank1));

  // Test Case: 2
  TriFilterbank triflt2(23, 33, 8000, 300, 3700, FrequencyScale::MEL);
  std::vector<float> input2{
      0.0461713, 0.0971317, 0.823457, 0.694828, 0.317099, 0.950222, 0.0344460,
      0.438744,  0.381558,  0.765516, 0.795199, 0.186872, 0.489764, 0.445586,
      0.646313,  0.709364,  0.754686, 0.276025, 0.679702, 0.655098, 0.162611,
      0.118997,  0.498364,  0.959743, 0.340385, 0.585267, 0.223811, 0.751267,
      0.255095,  0.505957,  0.699076, 0.890903, 0.959291};
  std::vector<float> matlabop2{
      0.578693, 0.131362, 0.301871, 0.426760, 0.523461, 0.0338169,
      0.285265, 0.311304, 0.424245, 0.714087, 0.680402, 0.267582,
      0.526783, 0.612373, 0.814208, 0.962699, 0.620225, 0.907083,
      0.326320, 0.879130, 1.07004,  0.844134, 0.957356};

  auto output2 = triflt2.apply(input2);
  // Implementation should match with matlab for Test case 2.
  ASSERT_TRUE(compareVec<float>(output2, matlabop2));
}

TEST(TriFilterbankTest, batchingTest) {
  int numFilters = 16, filterLen = 10, B = 15;
  auto input = randVec<float>(filterLen * B);
  auto triflt = TriFilterbank(numFilters, filterLen, 16000);
  auto output = triflt.apply(input);
  ASSERT_EQ(output.size(), numFilters * B);
  for (int i = 0; i < B; ++i) {
    std::vector<float> curInput(filterLen), expOutput(numFilters);
    std::copy(
        input.data() + i * filterLen,
        input.data() + (i + 1) * filterLen,
        curInput.data());
    std::copy(
        output.data() + i * numFilters,
        output.data() + (i + 1) * numFilters,
        expOutput.data());
    auto curOutput = triflt.apply(curInput);
    ASSERT_TRUE(compareVec<float>(curOutput, expOutput, 1E-5));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
