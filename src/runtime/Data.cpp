/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>

#include "data/W2lBlobsDataset.h"
#include "data/W2lListFilesDataset.h"
#include "runtime/Data.h"

#ifdef W2L_BUILD_FB_DEPENDENCIES
#include "fb/W2lEverstoreDataset.h"
#endif

namespace w2l {

std::shared_ptr<W2lDataset> createDataset(
    const std::string& path,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon /* = LexiconMap() */,
    int batchSize /* = 1 */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */,
    bool fallback2Ltr /* = true */,
    bool skipUnk /* = true */) {
  std::shared_ptr<W2lDataset> ds;
  if (FLAGS_everstoredb) {
#ifdef W2L_BUILD_FB_DEPENDENCIES
    W2lEverstoreDataset::init(); // Required for everstore client
    ds = std::make_shared<W2lEverstoreDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir,
        FLAGS_use_memcache);
#else
    LOG(FATAL) << "W2lEverstoreDataset not supported: "
               << "build with -DW2L_BUILD_FB_DEPENDENCIES";
#endif
  } else if (FLAGS_blobdata) {
    ds = std::make_shared<W2lBlobsDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir);
  } else {
    ds = std::make_shared<W2lListFilesDataset>(
        path,
        dicts,
        lexicon,
        batchSize,
        worldRank,
        worldSize,
        fallback2Ltr,
        skipUnk,
        FLAGS_datadir);
  }

  return ds;
}

std::shared_ptr<fl::Dataset> loadDataset(
    const std::vector<std::string>& paths,
    const std::string& rootDir /*  = "" */,
    int64_t batchSize /* = 1 */,
    int64_t worldRank /* = 0 */,
    int64_t worldSize /* = 1 */,
    const fl::Dataset::DataTransformFunction& inputTransform /*  = nullptr */,
    const fl::Dataset::DataTransformFunction& targetTransform /* = nullptr */) {
  std::vector<std::shared_ptr<const fl::Dataset>> listDs;
  std::vector<double> sizes;
  for (auto& path : paths) {
    auto lsDs = std::make_shared<ListFileDataset>(
        pathsConcat(rootDir, path), inputTransform, targetTransform);
    listDs.emplace_back(lsDs);
    const auto& curSizes = lsDs->getSampleSizes();
    sizes.insert(sizes.end(), curSizes.begin(), curSizes.end());
  }

  // Order Dataset
  std::vector<int64_t> sortedIds(sizes.size());
  std::iota(sortedIds.begin(), sortedIds.end(), 0);
  auto cmp = [&sizes](const int64_t& l, const int64_t& r) {
    return sizes[l] < sizes[r];
  };
  std::stable_sort(sortedIds.begin(), sortedIds.end(), cmp);

  auto mergedDs = std::make_shared<fl::MergeDataset>(listDs);

  auto sortedDs = std::make_shared<fl::ResampleDataset>(mergedDs, sortedIds);

  // Partition the dataset and distribute
  auto partitions = fl::partitionByRoundRobin(
      sortedDs->size(), worldRank, worldSize, batchSize);
  auto paritionDs = std::make_shared<fl::ResampleDataset>(sortedDs, partitions);

  // Batch the dataset
  using fl::join;
  return std::make_shared<fl::BatchDataset>(
      paritionDs,
      batchSize,
      fl::BatchDatasetPolicy::INCLUDE_LAST,
      std::vector<fl::Dataset::BatchFunction>{
          [](const std::vector<af::array>& arr) { return join(arr, 0, 3); },
          [](const std::vector<af::array>& arr) { return join(arr, -1, 1); },
          [](const std::vector<af::array>& arr) { return join(arr, 0, 1); },
          [](const std::vector<af::array>& arr) { return join(arr, 0, 1); }});
}
} // namespace w2l
