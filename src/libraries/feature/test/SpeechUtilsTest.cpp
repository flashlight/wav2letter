/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arrayfire.h>
#include <gtest/gtest.h>

#include "TestUtils.h"
#include "libraries/feature/SpeechUtils.h"

using namespace w2l;

TEST(SpeechUtilsTest, SimpleMatmul) {
  /*
    A                B
    [ 2  3  4 ]       [ 2  3 ]
    [ 3  4  5 ],      [ 3  4 ]
    [ 4  5  6 ],      [ 4  5 ]
    [ 5  6  7 ],
  */
  std::vector<float> A = {2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 7};
  std::vector<float> B = {2, 3, 3, 4, 4, 5};
  auto op = cblasGemm(A, B, 2, 3);
  std::vector<float> expectedOp = {29, 38, 38, 50, 47, 62, 56, 74};
  EXPECT_TRUE(compareVec(op, expectedOp, 1E-10));
}

TEST(SpeechUtilsTest, AfMatmulCompare) {
  int numTests = 1000;
  auto afToVec = [](const af::array& arr) {
    std::vector<float> vec(arr.elements());
    arr.host(vec.data());
    return vec;
  };
  while (numTests--) {
    int m = (rand() % 64) + 1;
    int n = (rand() % 128) + 1;
    int k = (rand() % 256) + 1;
    // Note: Arrayfire is column major
    af::array a = af::randu(k, m);
    af::array b = af::randu(n, k);
    af::array c = af::matmul(a, b, AF_MAT_TRANS, AF_MAT_TRANS).T();
    auto aVec = afToVec(a);
    auto bVec = afToVec(b);
    auto cVec = cblasGemm(aVec, bVec, n, k);
    ASSERT_TRUE(compareVec(cVec, afToVec(c), 1E-4));
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
