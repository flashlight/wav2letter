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

#include <future>
#include <memory>

#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "libraries/common/Dictionary.h"
#include "libraries/common/WordUtils.h"

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

static std::function<int(void)> makeSucceedsAfterIters(int iters) {
  auto state = std::make_shared<int>(0);
  return [state, iters]() {
    if (++*state >= iters) {
      return 42;
    } else {
      throw std::runtime_error("bleh");
    }
  };
}

static std::function<int(void)> makeSucceedsAfterMs(double ms) {
  using namespace std::chrono;
  auto state = std::make_shared<time_point<steady_clock>>();
  return [state, ms]() {
    auto now = steady_clock::now();
    if (state->time_since_epoch().count() == 0) {
      *state = now;
    }
    if (now - *state >= duration<double, std::milli>(ms)) {
      return 42;
    } else {
      throw std::runtime_error("bleh");
    }
  };
}

template <class Fn>
std::future<typename std::result_of<Fn()>::type> retryAsync(
    std::chrono::duration<double> initial,
    double factor,
    int64_t iters,
    Fn f) {
  return std::async(std::launch::async, [=]() {
    return retryWithBackoff(initial, factor, iters, f);
  });
}

TEST(W2lCommonTest, RetryWithBackoff) {
  auto alwaysSucceeds = []() { return 42; };
  auto alwaysFails = []() -> int { throw std::runtime_error("bleh"); };

  std::vector<std::future<int>> goods;
  std::vector<std::future<int>> bads;
  std::vector<std::future<int>> invalids;

  auto ms0 = std::chrono::milliseconds(0);
  auto ms50 = std::chrono::milliseconds(50);

  goods.push_back(retryAsync(ms0, 1.0, 5, alwaysSucceeds));
  goods.push_back(retryAsync(ms50, 2.0, 5, alwaysSucceeds));

  bads.push_back(retryAsync(ms0, 1.0, 5, alwaysFails));
  bads.push_back(retryAsync(ms50, 2.0, 5, alwaysFails));

  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterIters(6)));
  bads.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterIters(6)));
  goods.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterIters(5)));
  goods.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterIters(5)));

  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterMs(999)));
  bads.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterMs(999)));
  bads.push_back(retryAsync(ms0, 1.0, 5, makeSucceedsAfterMs(500)));
  goods.push_back(retryAsync(ms50, 2.0, 5, makeSucceedsAfterMs(500)));

  invalids.push_back(retryAsync(-ms50, 2.0, 5, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, -1.0, 5, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, 2.0, 0, alwaysSucceeds));
  invalids.push_back(retryAsync(ms50, 2.0, -1, alwaysSucceeds));

  for (auto& fut : goods) {
    ASSERT_EQ(fut.get(), 42);
  }
  for (auto& fut : bads) {
    ASSERT_THROW(fut.get(), std::runtime_error);
  }
  for (auto& fut : invalids) {
    ASSERT_THROW(fut.get(), std::invalid_argument);
  }

  // check special case promise<void> / future<void>
  auto alwaysSucceedsVoid = []() -> void {};
  auto alwaysFailsVoid = []() -> void { throw std::runtime_error("bleh"); };

  retryAsync(ms0, 1.0, 5, alwaysSucceedsVoid).get();
  ASSERT_THROW(
      retryAsync(ms0, 1.0, 5, alwaysFailsVoid).get(), std::runtime_error);
}

TEST(W2lCommonTest, PackReplabels) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);

  std::vector<int> labels = {5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10};
  std::vector<std::vector<int>> packedCheck(4);
  packedCheck[0] = {5, 6, 6, 6, 10, 8, 8, 10, 10, 10, 10, 10};
  packedCheck[1] = {5, 6, 1, 6, 10, 8, 1, 10, 1, 10, 1, 10};
  packedCheck[2] = {5, 6, 2, 10, 8, 1, 10, 2, 10, 1};
  packedCheck[3] = {5, 6, 2, 10, 8, 1, 10, 3, 10};

  for (int i = 0; i <= 3; ++i) {
    auto packed = packReplabels(labels, dict, i);
    ASSERT_EQ(packed, packedCheck[i]);
    auto unpacked = unpackReplabels(packed, dict, i);
    ASSERT_EQ(unpacked, labels);
  }
}

TEST(W2lCommonTest, Dictionary) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);
  dict.addEntry("4", 3);

  ASSERT_EQ(dict.getEntry(1), "1");
  ASSERT_EQ(dict.getEntry(3), "3");

  ASSERT_EQ(dict.getIndex("2"), 2);
  ASSERT_EQ(dict.getIndex("4"), 3);

  ASSERT_EQ(dict.entrySize(), 4);
  ASSERT_EQ(dict.indexSize(), 3);

  dict.addEntry("5");
  ASSERT_EQ(dict.getIndex("5"), 4);
  ASSERT_EQ(dict.entrySize(), 5);

  dict.addEntry("6");
  ASSERT_EQ(dict.getIndex("6"), 5);
  ASSERT_EQ(dict.indexSize(), 5);
}

TEST(W2lCommonTest, UnpackReplabels) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);
  dict.addEntry("3", 3);
  std::vector<int> labels = {6, 3, 7, 2, 8, 0, 1};

  auto unpacked1 = unpackReplabels(labels, dict, 1);
  ASSERT_THAT(unpacked1, ::testing::ElementsAre(6, 3, 7, 2, 8, 0, 0));

  auto unpacked2 = unpackReplabels(labels, dict, 2);
  ASSERT_THAT(unpacked2, ::testing::ElementsAre(6, 3, 7, 7, 7, 8, 0, 0));

  auto unpacked3 = unpackReplabels(labels, dict, 3);
  ASSERT_THAT(unpacked3, ::testing::ElementsAre(6, 6, 6, 6, 7, 7, 7, 8, 0, 0));
}

TEST(W2lCommonTest, UnpackReplabelsIgnoresInvalid) {
  Dictionary dict;
  dict.addEntry("1", 1);
  dict.addEntry("2", 2);

  // The initial replabel "1", with no prior token to repeat, is ignored.
  std::vector<int> labels1 = {1, 5, 1, 6};
  auto unpacked1 = unpackReplabels(labels1, dict, 2);
  ASSERT_THAT(unpacked1, ::testing::ElementsAre(5, 5, 6));

  // The final replabel "2", whose prior token is a replabel, is ignored.
  std::vector<int> labels2 = {1, 5, 1, 2, 6};
  auto unpacked2 = unpackReplabels(labels2, dict, 2);
  ASSERT_THAT(unpacked2, ::testing::ElementsAre(5, 5, 6));
  // With maxReps=1, "2" is not considered a replabel, altering the result.
  auto unpacked2_1 = unpackReplabels(labels2, dict, 1);
  ASSERT_THAT(unpacked2_1, ::testing::ElementsAre(5, 5, 2, 6));

  // All replabels past the first "1" are ignored here.
  std::vector<int> labels3 = {5, 1, 2, 1, 2, 6};
  auto unpacked3 = unpackReplabels(labels3, dict, 2);
  ASSERT_THAT(unpacked3, ::testing::ElementsAre(5, 5, 6));
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

TEST(W2lCommonTest, AfMatrixToStrings) {
  std::vector<int> arr = {119, 97,  118, -1,  -1,  -1,  -1,  -1, -1, -1, -1,
                          -1,  108, 101, 116, 116, 101, 114, -1, -1, -1};
  af::array afArr(6, 3, arr.data());
  auto stringVec = afMatrixToStrings<int>(afArr, -1);
  ASSERT_EQ(stringVec.size(), 3);
  ASSERT_EQ(stringVec[0], "wav");
  ASSERT_EQ(stringVec[1], "");
  ASSERT_EQ(stringVec[2], "letter");
}

TEST(W2lCommonTest, WrdToTarget) {
  gflags::FlagSaver flagsaver;
  w2l::FLAGS_wordseparator = "_";

  LexiconMap lexicon;
  // word pieces with word separator in the end
  lexicon["123"].push_back({"1", "23_"});
  lexicon["456"].push_back({"456_"});
  // word pieces with word separator in the beginning
  lexicon["789"].push_back({"_7", "89"});
  lexicon["010"].push_back({"_0", "10"});
  // word pieces without word separators
  lexicon["105"].push_back({"10", "5"});
  lexicon["2100"].push_back({"2", "1", "00"});
  // letters
  lexicon["888"].push_back({"8", "8", "8"});
  lexicon["12"].push_back({"1", "2"});
  lexicon[kUnkToken] = {};

  Dictionary dict;
  for (auto l : lexicon) {
    for (auto p : l.second) {
      for (auto c : p) {
        if (!dict.contains(c)) {
          dict.addEntry(c);
        }
      }
    }
  }
  dict.addEntry("_");

  std::vector<std::string> words = {"123", "456"};
  auto target = wrd2Target(words, lexicon, dict);
  ASSERT_THAT(target, ::testing::ElementsAreArray({"1", "23_", "456_"}));

  std::vector<std::string> words1 = {"789", "010"};
  auto target1 = wrd2Target(words1, lexicon, dict);
  ASSERT_THAT(target1, ::testing::ElementsAreArray({"_7", "89", "_0", "10"}));

  std::vector<std::string> words2 = {"105", "2100"};
  auto target2 = wrd2Target(words2, lexicon, dict);
  ASSERT_THAT(
      target2, ::testing::ElementsAreArray({"10", "5", "_", "2", "1", "00"}));

  std::vector<std::string> words3 = {"12", "888", "12"};
  auto target3 = wrd2Target(words3, lexicon, dict);
  ASSERT_THAT(
      target3,
      ::testing::ElementsAreArray(
          {"1", "2", "_", "8", "8", "8", "_", "1", "2"}));

  // unknown words "111", "199"
  std::vector<std::string> words4 = {"111", "789", "199"};
  // fall back to letters and skip unknown
  auto target4 = wrd2Target(words4, lexicon, dict, true, true);
  ASSERT_THAT(
      target4,
      ::testing::ElementsAreArray({"1", "1", "1", "_7", "89", "_", "1"}));
  // skip unknown
  target4 = wrd2Target(words4, lexicon, dict, false, true);
  ASSERT_THAT(target4, ::testing::ElementsAreArray({"_7", "89"}));
}

TEST(W2lCommonTest, TargetToSingleLtr) {
  gflags::FlagSaver flagsaver;
  w2l::FLAGS_wordseparator = "_";
  w2l::FLAGS_usewordpiece = true;

  Dictionary dict;
  for (int i = 0; i < 10; ++i) {
    dict.addEntry(std::to_string(i), i);
  }
  dict.addEntry("_", 10);
  dict.addEntry("23_", 230);
  dict.addEntry("456_", 4560);

  std::vector<int> words = {1, 230, 4560};
  auto target = tknIdx2Ltr(words, dict);
  ASSERT_THAT(
      target, ::testing::ElementsAreArray({"1", "2", "3", "_", "4", "5", "6"}));
}

TEST(W2lCommonTest, UT8Split) {
  // ASCII
  std::string in1 = "Vendetta";
  auto in1Tkns = splitWrd(in1);
  for (int i = 0; i < in1.size(); ++i) {
    ASSERT_EQ(std::string(1, in1[i]), in1Tkns[i]);
  }

  // NFKC encoding
  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::string in2 = "Beyoncé";
  auto in2Tkns = splitWrd(in2);

  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::vector<std::string> in2TknsExp = {"B", "e", "y", "o", "n", "c", "é"};
  ASSERT_EQ(in2Tkns.size(), 7);
  for (int i = 0; i < in2Tkns.size(); ++i) {
    ASSERT_EQ(in2TknsExp[i], in2Tkns[i]);
  }

  // NFKD encoding
  // @lint-ignore TXT5 Source code should only include printable US-ASCII bytes.
  std::string in3 = "Beyoncé";
  auto in3Tkns = splitWrd(in3);
  std::vector<std::string> in3TknsExp = {
      "B", "e", "y", "o", "n", "c", "e", u8"\u0301"};
  ASSERT_EQ(in3Tkns.size(), 8);
  for (int i = 0; i < in3Tkns.size(); ++i) {
    ASSERT_EQ(in3TknsExp[i], in3Tkns[i]);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
