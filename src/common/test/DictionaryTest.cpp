/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdexcept>
#include <string>

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "common/FlashlightUtils.h"
#include "libraries/common/Dictionary.h"

using namespace w2l;

std::string loadPath = "";

TEST(DictionaryTest, TestBasic) {
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

TEST(DictionaryTest, FromFile) {
  ASSERT_THROW(Dictionary("not_a_real_file"), std::invalid_argument);

  Dictionary dict(pathsConcat(loadPath, "test.dict"));
  ASSERT_EQ(dict.entrySize(), 10);
  ASSERT_EQ(dict.indexSize(), 7);
  ASSERT_TRUE(dict.contains("a"));
  ASSERT_FALSE(dict.contains("q"));
  ASSERT_EQ(dict.getEntry(1), "b");
  ASSERT_EQ(dict.getIndex("e"), 4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for sample dictionary
#ifdef W2L_DICTIONARY_TEST_DIR
  loadPath = W2L_DICTIONARY_TEST_DIR;
#endif

  return RUN_ALL_TESTS();
}
