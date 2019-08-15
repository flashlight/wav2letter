/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <flashlight/flashlight.h>

#include "common/Utils.h"
#include "data/ListFileDataset.h"
#include "data/W2lDataset.h"
#include "libraries/common/Dictionary.h"

namespace w2l {

std::shared_ptr<fl::Dataset> loadDataset(
    const std::vector<std::string>& paths,
    const std::string& rootDir = "",
    int64_t batchSize = 1,
    int64_t worldRank = 0,
    int64_t worldSize = 1,
    const fl::Dataset::DataTransformFunction& inputTransform = nullptr,
    const fl::Dataset::DataTransformFunction& targetTransform = nullptr);

std::shared_ptr<W2lDataset> createDataset(
    const std::string& path,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon = LexiconMap(),
    int batchSize = 1,
    int worldRank = 0,
    int worldSize = 1,
    bool fallback2Ltr = true,
    bool skipUnk = true);

} // namespace w2l
