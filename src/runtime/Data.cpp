/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <glog/logging.h>

#include "data/W2lListFilesDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "runtime/Data.h"

#ifdef BUILD_FB_DEPENDENCIES
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
#ifdef BUILD_FB_DEPENDENCIES
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
        FLAGS_datadir);
#else
    LOG(FATAL) << "W2lEverstoreDataset not supported: "
               << "build with -DBUILD_FB_DEPENDENCIES";
#endif
  } else if (FLAGS_listdata) {
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
  } else {
    ds = std::make_shared<W2lNumberedFilesDataset>(
        path, dicts, batchSize, worldRank, worldSize, FLAGS_datadir);
  }

  return ds;
}
} // namespace w2l
