/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "libraries/common/Dictionary.h"

namespace w2l {

using LexiconMap =
    std::unordered_map<std::string, std::vector<std::vector<std::string>>>;

Dictionary createWordDict(const LexiconMap& lexicon);

LexiconMap loadWords(const std::string& filename, int maxWords = -1);

// split word into tokens abc -> {"a", "b", "c"}
// Works with ASCII, UTF-8 encodings
std::vector<std::string> splitWrd(const std::string& word);

std::vector<int> tkn2Idx(
    const std::vector<std::string>& spelling,
    const Dictionary& tokenDict,
    int maxReps);

std::vector<int> validateIdx(std::vector<int> input, int unkIdx);

/**
 * Pack a token sequence by replacing consecutive repeats with replabels,
 * e.g. "abbccc" -> "ab1c2". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
std::vector<int> packReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);

/**
 * Unpack a token sequence by replacing replabels with repeated tokens,
 * e.g. "ab1c2" -> "abbccc". The tokens "1", "2", ..., `to_string(maxReps)`
 * must already be in `dict`.
 */
std::vector<int> unpackReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps);

} // namespace w2l
