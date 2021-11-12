// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <gtest/gtest.h>
#include <filesystem>
#include <string>
#include "deeplearning/projects/wav2letter/recipes/joint_training_vox_populi/cpc/MTLLoss.h"

using namespace ::testing;
namespace fs = std::filesystem;
typedef std::map<std::string, unsigned int> Mapping;

TEST(test_mtl, load_mapping) {
  const fs::path ref_dir = fs::path(__FILE__).parent_path();
  const std::string data_path = ref_dir / "test_mapping.txt";
  Mapping test_mapping = asr4real::loadMapping(data_path);
  Mapping expected = Mapping();
  expected["en"] = 0;
  expected["af"] = 1;
  expected["eu"] = 2;
  expected["ro"] = 3;
  expected["it"] = 4;
  EXPECT_EQ(test_mapping.size(), 5);
  EXPECT_EQ(test_mapping, expected);
}

TEST(test_mtl, get_map_index) {
  const fs::path ref_dir = fs::path(__FILE__).parent_path();
  const std::string data_path = ref_dir / "test_mapping.txt";
  const Mapping test_mapping = asr4real::loadMapping(data_path);

  EXPECT_EQ(asr4real::getMapIndexFromFileID("fr#en", test_mapping), 0);
  EXPECT_EQ(asr4real::getMapIndexFromFileID("fromage#ro", test_mapping), 3);
  EXPECT_EQ(asr4real::getMapIndexFromFileID("af#it", test_mapping), 4);
}
