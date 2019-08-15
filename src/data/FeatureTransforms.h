/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <flashlight/flashlight.h>

#include "libraries/common/WordUtils.h"
#include "libraries/feature/FeatureParams.h"

namespace w2l {

fl::Dataset::DataTransformFunction inputFeatures(const FeatureParams& params);

fl::Dataset::DataTransformFunction targetFeatures(
    const Dictionary& dict,
    const LexiconMap& lexicon,
    int wordSeperator,
    bool surround = true);

} // namespace w2l
