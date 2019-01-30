/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <memory>

#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"

using namespace w2l;

TEST(W2lCommonTest, StringTrim) {
  EXPECT_EQ(trim(""), "");
  EXPECT_EQ(trim("     "), "");
  EXPECT_EQ(trim(" \n \tabc   "), "abc");
  EXPECT_EQ(trim("ab     ."), "ab     .");
  EXPECT_EQ(trim("|   ab cd   "), "|   ab cd");
}

TEST(W2lCommonTest, ReplaceAll) {
  std::string in = "\tSomewhere, something incredible is waiting to be known.";
  replaceAll(in, "\t", "   ");
  EXPECT_EQ(in, "   Somewhere, something incredible is waiting to be known.");
  replaceAll(in, "   ", "");
  EXPECT_EQ(in, "Somewhere, something incredible is waiting to be known.");
  replaceAll(in, "some", "any");
  EXPECT_EQ(in, "Somewhere, anything incredible is waiting to be known.");
  replaceAll(in, " ", "");
  EXPECT_EQ(in, "Somewhere,anythingincredibleiswaitingtobeknown.");
}

TEST(W2lCommonTest, StringSplit) {
  using Pieces = std::vector<std::string>;
  const std::string& input = " ;abc; de;;";
  // char delimiters
  EXPECT_EQ(split(';', input), (Pieces{" ", "abc", " de", "", ""}));
  EXPECT_EQ(split(';', input, true), (Pieces{" ", "abc", " de"}));
  EXPECT_EQ(split('X', input), (Pieces{input}));
  // string delimiters
  EXPECT_EQ(split(";;", input), (Pieces{" ;abc; de", ""}));
  EXPECT_EQ(split(";;", input, true), (Pieces{" ;abc; de"}));
  EXPECT_EQ(split("ac", input), (Pieces{input}));
  EXPECT_THROW(split("", input), std::invalid_argument);
  // multi-char delimiters
  EXPECT_EQ(splitOnAnyOf("bce", input), (Pieces{" ;a", "", "; d", ";;"}));
  EXPECT_EQ(splitOnAnyOf("bce", input, true), (Pieces{" ;a", "; d", ";;"}));
  EXPECT_EQ(splitOnAnyOf("", input), (Pieces{input}));
  // whitespace
  EXPECT_EQ(splitOnWhitespace(input), (Pieces{"", ";abc;", "de;;"}));
  EXPECT_EQ(splitOnWhitespace(input, true), (Pieces{";abc;", "de;;"}));
}

TEST(W2lCommonTest, StringJoin) {
  using Pieces = std::vector<std::string>;
  // from vector
  EXPECT_EQ(join("", Pieces{"a", "b", "", "c"}), "abc");
  EXPECT_EQ(join(",", Pieces{"a", "b", "", "c"}), "a,b,,c");
  EXPECT_EQ(join(",\n", Pieces{"a", "b", "", "c"}), "a,\nb,\n,\nc");
  EXPECT_EQ(join("abc", Pieces{}), "");
  EXPECT_EQ(join("abc", Pieces{"abc"}), "abc");
  EXPECT_EQ(join("abc", Pieces{"abc", "abc"}), "abcabcabc");
  // from iterator range
  Pieces input{"in", "te", "re", "st", "ing"};
  EXPECT_EQ(join("", input.begin(), input.end()), "interesting");
  EXPECT_EQ(join("", input.begin(), input.end() - 1), "interest");
  EXPECT_EQ(join("", input.begin(), input.begin()), "");
  EXPECT_EQ(join("e", input.begin() + 1, input.end() - 1), "teereest");
}

TEST(W2lCommonTest, StringFormat) {
  EXPECT_EQ(format("a%sa", "bbb"), "abbba");
  EXPECT_EQ(format("%%%c%s%c", 'a', "bbb", 'c'), "%abbbc");
  EXPECT_EQ(format("0x%08x", 0x0023ffaa), "0x0023ffaa");
  EXPECT_EQ(format("%5s", "abc"), "  abc");
  EXPECT_EQ(format("%.3f", 3.1415926), "3.142");

  std::string big(2000, 'a');
  EXPECT_EQ(format("(%s)", big.c_str()), std::string() + "(" + big + ")");
}

TEST(W2lCommonTest, PathsConcat) {
  auto path1 = pathsConcat("/tmp/", "test.wav");
  auto path2 = pathsConcat("/tmp", "test.wav");
  ASSERT_EQ(path1, "/tmp/test.wav");
  ASSERT_EQ(path2, "/tmp/test.wav");
}

TEST(W2lCommonTest, Replabel) {
  Dictionary dict;
  dict.addToken("1", 1);
  dict.addToken("2", 2);
  dict.addToken("3", 3);
  std::vector<int> lab = {5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10};

  auto lab0 = lab;
  replaceReplabels(lab0, 0, dict);
  ASSERT_THAT(
      lab0,
      ::testing::ElementsAreArray({5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10}));
  invReplaceReplabels(lab0, 0, dict);
  ASSERT_THAT(lab0, ::testing::ElementsAreArray(lab));

  auto lab1 = lab;
  replaceReplabels(lab1, 1, dict);
  ASSERT_THAT(
      lab1,
      ::testing::ElementsAreArray({5, 6, 1, 6, 10, 8, 1, 10, 1, 10, 1, 10}));
  invReplaceReplabels(lab1, 1, dict);
  ASSERT_THAT(lab1, ::testing::ElementsAreArray(lab));

  auto lab2 = lab;
  replaceReplabels(lab2, 2, dict);
  ASSERT_THAT(
      lab2, ::testing::ElementsAreArray({5, 6, 2, 10, 8, 1, 10, 2, 10, 1}));
  invReplaceReplabels(lab2, 2, dict);
  ASSERT_THAT(lab2, ::testing::ElementsAreArray(lab));

  auto lab3 = lab;
  replaceReplabels(lab3, 3, dict);
  ASSERT_THAT(
      lab3, ::testing::ElementsAreArray({5, 6, 2, 10, 8, 1, 10, 3, 10}));
  invReplaceReplabels(lab3, 3, dict);
  ASSERT_THAT(lab3, ::testing::ElementsAreArray(lab));
}

TEST(W2lCommonTest, Dictionary) {
  Dictionary dict;
  dict.addToken("1", 1);
  dict.addToken("2", 2);
  dict.addToken("3", 3);
  dict.addToken("4", 3);

  ASSERT_EQ(dict.getToken(1), "1");
  ASSERT_EQ(dict.getToken(3), "3");

  ASSERT_EQ(dict.getIndex("2"), 2);
  ASSERT_EQ(dict.getIndex("4"), 3);

  ASSERT_EQ(dict.tokenSize(), 4);
  ASSERT_EQ(dict.indexSize(), 3);

  dict.addToken("5");
  ASSERT_EQ(dict.getIndex("5"), 4);
  ASSERT_EQ(dict.tokenSize(), 5);

  dict.addToken("6");
  ASSERT_EQ(dict.getIndex("6"), 5);
  ASSERT_EQ(dict.indexSize(), 5);
}

TEST(W2lCommonTest, InvReplabel) {
  Dictionary dict;
  dict.addToken("1", 1);
  dict.addToken("2", 2);
  dict.addToken("3", 3);
  std::vector<int> lab = {6, 3, 7, 2, 8, 0, 1};

  auto lab1 = lab;
  invReplaceReplabels(lab1, 1, dict);
  ASSERT_THAT(lab1, ::testing::ElementsAre(6, 3, 7, 2, 8, 0, 0));

  auto lab2 = lab;
  invReplaceReplabels(lab2, 2, dict);
  ASSERT_THAT(lab2, ::testing::ElementsAre(6, 3, 7, 7, 7, 8, 0, 0));

  auto lab3 = lab;
  invReplaceReplabels(lab3, 3, dict);
  ASSERT_THAT(lab3, ::testing::ElementsAre(6, 6, 6, 6, 7, 7, 7, 8, 0, 0));
}

TEST(W2lCommonTest, Uniq) {
  std::vector<int> uq1 = {5, 6, 6, 8, 9, 8, 8, 8};
  uniq(uq1);
  ASSERT_THAT(uq1, ::testing::ElementsAre(5, 6, 8, 9, 8));

  std::vector<int> uq2 = {1, 1, 1, 1, 1};
  uniq(uq2);
  ASSERT_THAT(uq2, ::testing::ElementsAre(1));
}

TEST(W2lCommonTest, Normalize) {
  double threshold = 0.01;
  auto afNormalize = [threshold](const af::array& in, int batchdim) {
    auto elementsPerBatch = in.elements() / in.dims(batchdim);
    auto in2d = af::moddims(in, elementsPerBatch, in.dims(batchdim));

    af::array meandiff = (in2d - af::tile(af::mean(in2d, 0), elementsPerBatch));

    af::array stddev = af::stdev(in2d, 0);
    af::replace(stddev, stddev > threshold, 1.0);

    return af::moddims(
        meandiff / af::tile(stddev, elementsPerBatch), in.dims());
  };
  auto arr = af::randu(13, 17, 19);
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());

  auto arrVecNrm = normalize(arrVec, 19, threshold);
  auto arrNrm = af::array(arr.dims(), arrVecNrm.data());
  ASSERT_TRUE(af::allTrue<bool>(af::abs(arrNrm - afNormalize(arr, 2)) < 1E-5));
}

TEST(W2lCommonTest, Transpose) {
  auto arr = af::randu(13, 17, 19, 23);
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());
  auto arrVecT = transpose2d<float>(arrVec, 17, 13, 19 * 23);
  auto arrT = af::array(17, 13, 19, 23, arrVecT.data());
  ASSERT_TRUE(af::allTrue<bool>(arrT - arr.T() == 0.0));
}

TEST(W2lCommonTest, localNormalize) {
  auto afNormalize = [](const af::array& in, int64_t lw, int64_t rw) {
    auto out = in;
    for (int64_t b = 0; b < in.dims(3); ++b) {
      for (int64_t i = 0; i < in.dims(0); ++i) {
        int64_t b_idx = (i - lw > 0) ? (i - lw) : 0;
        int64_t e_idx = (in.dims(0) - 1 > i + rw) ? (i + rw) : (in.dims(0) - 1);

        // Need to call af::flat because of some weird bug in Arrayfire
        af::array slice = in(af::seq(b_idx, e_idx), af::span, af::span, b);
        auto mean = af::mean<float>(af::flat(slice));
        auto stddev = af::stdev<float>(af::flat(slice));

        out(i, af::span, af::span, b) -= mean;
        if (stddev > 0.0) {
          out(i, af::span, af::span, b) /= stddev;
        }
      }
    }
    return out;
  };
  auto arr = af::randu(47, 67, 2, 10); // FRAMES X FEAT X CHANNELS X BATCHSIZE
  std::vector<float> arrVec(arr.elements());
  arr.host(arrVec.data());

  std::vector<std::pair<int, int>> ctx = {
      {0, 0}, {1, 1}, {2, 2}, {4, 4}, {1024, 1024}, {10, 0}, {2, 12}};

  for (auto c : ctx) {
    auto arrVecNrm = localNormalize(
        arrVec,
        c.first /* context */,
        c.second,
        arr.dims(0) /* frames */,
        arr.dims(3) /*batches */);
    auto arrNrm = af::array(arr.dims(), arrVecNrm.data());
    ASSERT_TRUE(af::allTrue<bool>(
        af::abs(arrNrm - afNormalize(arr, c.first, c.second)) < 1E-4));
  }
}

TEST(W2lCommonTest, AfToVectorString) {
  std::vector<int> arr = {119, 97,  118, -1,  -1,  -1,  -1,  -1, -1, -1, -1,
                          -1,  108, 101, 116, 116, 101, 114, -1, -1, -1};
  af::array afArr(6, 3, arr.data());
  auto stringVec = afToVector<std::string>(afArr);
  ASSERT_EQ(stringVec.size(), 3);
  ASSERT_EQ(stringVec[0], "wav");
  ASSERT_EQ(stringVec[1], "");
  ASSERT_EQ(stringVec[2], "letter");
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
