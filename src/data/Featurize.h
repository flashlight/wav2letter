/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <unordered_map>

#include "data/NumberedFilesLoader.h"
#include "data/Sound.h"
#include "libraries/common/Dictionary.h"
#include "libraries/feature/FeatureParams.h"

namespace w2l {

typedef std::unordered_map<int, af::dim4> DimsMap;
typedef std::unordered_map<int, std::vector<int>> TargetFeatMap;

struct W2lFeatureData {
  std::vector<float> input;
  TargetFeatMap targets;
  af::dim4 inputDims;
  DimsMap targetDims;
  std::vector<int> sampleIds;
  af::dim4 sampleIdsDims;
};

W2lFeatureData featurize(
    const std::vector<W2lLoaderData>& data,
    const DictionaryMap& dicts);

FeatureParams defineSpeechFeatureParams();

int64_t getSpeechFeatureSize();

} // namespace w2l
