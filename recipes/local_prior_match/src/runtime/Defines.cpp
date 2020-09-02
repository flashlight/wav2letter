/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/runtime/Defines.h"

namespace w2l {

// data scheduler
DEFINE_string(trainaudio, "", "Unpaired audio training data");
DEFINE_int64(
    pairediter,
    1,
    "Number of steps per epoch for the paired training set");
DEFINE_int64(
    audioiter,
    0,
    "Number of steps per epoch for the unpaired audio training set");
DEFINE_string(
    schedulerorder,
    kUniformOrder,
    "the access order between the datasets in the data scheduler (uniform, inorder, random)");

// lm
DEFINE_string(lmdict, "", "Dictionary used in LM training");

// within-beam prior-match
DEFINE_int64(lpmBeamsz, 4, "Beam size for prior matching objective");
DEFINE_double(
    hyplenratiolb,
    0.95,
    "Discard hypotheses shorter than ref length multiplied by this. Set to <0 to deactivate");
DEFINE_double(
    hyplenratioub,
    1.05,
    "Discard hypotheses longer than ref length multiplied by this. Set to <0 to deactivate");
DEFINE_int64(unpairedBatchsize, 4, "Batch size for unpaired data");
DEFINE_string(
    proposalModel,
    "",
    "Path to load the proposal model for beam search.");
DEFINE_string(
    propupdate,
    kBetter,
    "Update rule for proposal model (never, always, better)");

} // namespace w2l
