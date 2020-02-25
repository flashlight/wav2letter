/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <utility>

#include <flashlight/flashlight.h>

#include "criterion/criterion.h"
#include "libraries/common/Dictionary.h"

namespace w2l {
/**
 * Generate mapping between the indices of tokens in dict1 and dict2
 * for matching the dictionaries in w2l and fairseq.
 * The function returns mapping, where the
 * token with index i in dict1 maps to the token with index mapping[i] in dict2.
 */
std::vector<int> genTokenDictIndexMap(
    const Dictionary& dict1,
    const Dictionary& dict2);

af::array getTargetLength(af::array& target, int eosIdx);

std::pair<std::vector<std::vector<int>>, std::vector<int>> batchBeamSearch(
    const fl::Variable& output,
    const std::shared_ptr<Seq2SeqCriterion>& criterion,
    int eos);

std::pair<std::vector<std::vector<int>>, std::vector<int>> filterBeamByLength(
    const std::vector<std::vector<int>>& paths,
    const std::vector<int>& hypoNums,
    const std::vector<int>& refLengths);

fl::Variable adjustProb(
    const fl::Variable& logprob,
    const std::vector<int>& hypoNums,
    bool renormalize,
    bool linear);

fl::Variable entropy(const fl::Variable& logprob);

af::array batchTarget(
    const std::vector<std::vector<int>>& tgt,
    const int& padVal);

fl::Variable batchEncoderOutput(
    const std::vector<int>& hypoNums,
    const fl::Variable& encoderOutput);

} // namespace w2l
