/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Generic utilities which should not depend on ArrayFire / flashlight.
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/Defines.h"
#include "libraries/common/Utils.h"
#include "libraries/common/WordUtils.h"

namespace w2l {

// ============================== Dataset helpers ==============================

// TODO: these should be cleaned up (de-FLAGS-ified) and
// moved to a more relevant location

std::vector<std::string> loadTarget(const std::string& filepath);

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

// ============================== Decoder helpers ==============================

// TODO: these should be cleaned up (de-FLAGS-ified) and
// moved to a more relevant location, probably libraries/common/WordUtils.h

/* A series of vector to vector mapping operations */

std::vector<std::string> tknIdx2Ltr(const std::vector<int>&, const Dictionary&);

std::vector<std::string> tkn2Wrd(const std::vector<std::string>&);

// will be deprecated soon
std::vector<std::string> wrdIdx2Wrd(const std::vector<int>&, const Dictionary&);

std::vector<std::string> tknTarget2Ltr(std::vector<int>, const Dictionary&);

std::vector<std::string> tknPrediction2Ltr(std::vector<int>, const Dictionary&);

} // namespace w2l
