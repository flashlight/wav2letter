/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <gflags/gflags.h>

#include "common/Defines.h"

namespace w2l {

// config
constexpr const char* kIteration = "iteration";
constexpr const char* kReloadPath = "reloadPath";
constexpr const char* kPropPath = "propPath";
constexpr const char* kRunStatus = "runStatus";
constexpr const char* kStartEpoch = "startEpoch";
constexpr const char* kStartIter = "startIter";

// meter
constexpr const char* kTarget = "L";
constexpr const char* kWord = "W";
constexpr const char* kASRLoss = "asr-loss";
constexpr const char* kLPMLoss = "lpm-loss";
constexpr const char* kFullLoss = "full-loss";
constexpr const char* kRuntime = "runtime";
constexpr const char* kTimer = "bch";
constexpr const char* kSampleTimer = "smp";
constexpr const char* kFwdTimer = "fwd";
constexpr const char* kCritFwdTimer = "crit-fwd";
constexpr const char* kBeamTimer = "beam";
constexpr const char* kBeamFwdTimer = "beam-fwd";
constexpr const char* kLMFwdTimer = "lm-fwd";
constexpr const char* kBwdTimer = "bwd";
constexpr const char* kOptimTimer = "optim";
constexpr const char* kNumHypos = "num-hypo";
constexpr const char* kLMEnt = "lm-ent";
constexpr const char* kLMScore = "lm-score";
constexpr const char* kLen = "len";

// data
// continue from src/common/Defines.h
constexpr size_t kDataTypeIdx = kNumDataIdx;
constexpr size_t kGlobalBatchIdx = kNumDataIdx + 1;
constexpr size_t kParallelData = 1;
constexpr size_t kUnpairedAudio = 2;
constexpr const char* kRandomOrder = "random";
constexpr const char* kInOrder = "inorder";
constexpr const char* kUniformOrder = "uniform";

// proposal-update type
constexpr const char* kNever = "never";
constexpr const char* kAlways = "always";
constexpr const char* kBetter = "better";

// flags
// data scheduler
DECLARE_string(trainaudio);
DECLARE_int64(pairediter);
DECLARE_int64(audioiter);
DECLARE_string(schedulerorder);
DECLARE_int64(unpairedBatchsize);

// lm
DECLARE_string(lmdict);

// within-beam prior matching
DECLARE_int64(lpmBeamsz);
DECLARE_double(hyplenratiolb);
DECLARE_double(hyplenratioub);
DECLARE_string(proposalModel);
DECLARE_string(propupdate);

} // namespace w2l
