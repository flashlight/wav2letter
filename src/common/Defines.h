/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <functional>
#include <unordered_map>

#define W2L_VERSION "0.1"

namespace w2l {

// Dataset indices
// If a new field is added, `kNumDataIdx` should be modified accordingly.
constexpr const size_t kInputIdx = 0;
constexpr const size_t kTargetIdx = 1;
constexpr const size_t kWordIdx = 2;
constexpr const size_t kFileIdIdx = 3;
constexpr const size_t kNumDataIdx = 4; // total number of dataset indices

// Various constants used in w2l
constexpr const char* kGflags = "gflags";
constexpr const char* kCommandLine = "commandline";
constexpr const char* kUserName = "username";
constexpr const char* kHostName = "hostname";
constexpr const char* kTimestamp = "timestamp";
constexpr const char* kRunIdx = "runIdx";
constexpr const char* kRunPath = "runPath";
constexpr const char* kProgramName = "programname";
constexpr const char* kEpoch = "epoch";
constexpr const char* kSGDoptimizer = "sgd";
constexpr const char* kAdamOptimizer = "adam";
constexpr const char* kRMSPropOptimizer = "rmsprop";
constexpr const char* kAdadeltaOptimizer = "adadelta";
constexpr const char* kCtcCriterion = "ctc";
constexpr const char* kAsgCriterion = "asg";
constexpr const char* kSeq2SeqCriterion = "seq2seq";
constexpr const char* kEosToken = "$";
constexpr const char* kBlankToken = "#";
constexpr const char* kSilToken = "|";
constexpr const char* kUnkToken = "<unk>";
constexpr const int kTryCatchAttempts = 5;
constexpr const int kTryCatchWaitSec = 5;
constexpr const int kTargetPadValue = -1;
constexpr const int kMaxDevicePerNode = 8;

// Feature params
constexpr const int kFrameSizeMs = 25;
constexpr const int kFrameStrideMs = 10;
constexpr const int kLifterParam = 22;
constexpr const int kPrefetchSize = 2;

template <class A, class B, class C>
std::function<C(A)> compose(std::function<B(A)> f, std::function<C(B)> g) {
  return [=](A a) -> C { return g(f(a)); };
}

template <class A, class B, class C>
std::function<C(A)> compose(std::function<B(A)> f, std::function<C(B&&)> g) {
  return [=](A a) -> C { return g(f(a)); };
}

/* ========== DATA OPTIONS ========== */

DECLARE_string(train);
DECLARE_string(valid);
DECLARE_string(test);
DECLARE_int64(batchsize);
DECLARE_string(input);
DECLARE_int64(samplerate);
DECLARE_int64(channels);
DECLARE_string(tokens);
DECLARE_int64(replabel);
DECLARE_string(surround);
DECLARE_bool(noresample);
DECLARE_bool(eostoken);
DECLARE_string(dataorder);
DECLARE_int64(inputbinsize);
DECLARE_int64(outputbinsize);

/* ========== FILTERING OPTIONS ========== */

DECLARE_bool(skipoov);
DECLARE_int64(minisz);
DECLARE_int64(maxisz);
DECLARE_int64(mintsz);
DECLARE_int64(maxtsz);

/* ========== NORMALIZATION OPTIONS ========== */

DECLARE_int64(localnrmlleftctx);
DECLARE_int64(localnrmlrightctx);
DECLARE_string(onorm);
DECLARE_bool(sqnorm);

/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */

DECLARE_int64(iter);
DECLARE_bool(itersave);
DECLARE_double(lr);
DECLARE_double(momentum);
DECLARE_double(weightdecay);
DECLARE_bool(sqnorm);
DECLARE_double(lrcrit);
DECLARE_double(maxgradnorm);
DECLARE_double(adambeta1);
DECLARE_double(adambeta2);
DECLARE_double(optimrho);
DECLARE_double(optimepsilon);

/* ========== LR-SCHEDULER OPTIONS ========== */

DECLARE_int64(stepsize);
DECLARE_double(gamma);

/* ========== OPTIMIZER OPTIONS ========== */
DECLARE_string(netoptim);
DECLARE_string(critoptim);

/* ========== MFCC OPTIONS ========== */

DECLARE_bool(mfcc);
DECLARE_bool(pow);
DECLARE_int64(mfcccoeffs);
DECLARE_bool(mfsc);
DECLARE_double(melfloor);
DECLARE_int64(filterbanks);
DECLARE_int64(devwin);
DECLARE_int64(fftcachesize);

/* ========== RUN OPTIONS ========== */

DECLARE_string(datadir);
DECLARE_string(tokensdir);
DECLARE_string(rundir);
DECLARE_string(archdir);
DECLARE_string(flagsfile);
DECLARE_string(runname);
DECLARE_int64(nthread);
DECLARE_string(tag);
DECLARE_int64(seed);
DECLARE_int64(memstepsize);
DECLARE_int64(reportiters);
DECLARE_int64(pcttraineval);

/* ========== ARCHITECTURE OPTIONS ========== */

DECLARE_string(arch);
DECLARE_string(criterion);
DECLARE_bool(garbage);
DECLARE_int64(encoderdim);

/* ========== DECODER OPTIONS ========== */

DECLARE_bool(show);
DECLARE_bool(showletters);
DECLARE_bool(forceendsil);
DECLARE_bool(logadd);

DECLARE_string(smearing);
DECLARE_string(lmtype);
DECLARE_string(lexicon);
DECLARE_string(emission_dir);
DECLARE_string(lm);
DECLARE_string(am);
DECLARE_string(sclite);

DECLARE_double(lmweight);
DECLARE_double(wordscore);
DECLARE_double(silweight);
DECLARE_double(unkweight);
DECLARE_double(beamscore);

DECLARE_int32(maxload);
DECLARE_int32(maxword);
DECLARE_int32(beamsize);
DECLARE_int32(nthread_decoder);

/* ========== ASG OPTIONS ========== */

DECLARE_int64(linseg);
DECLARE_double(linlr);
DECLARE_double(linlrcrit);
DECLARE_double(transdiag);

/* ========== SEQ2SEQ OPTIONS ========== */

DECLARE_int64(maxdecoderoutputlen);
DECLARE_int64(pctteacherforcing);
DECLARE_string(samplingstrategy);
DECLARE_double(labelsmooth);
DECLARE_bool(inputfeeding);
DECLARE_string(attention);
DECLARE_string(attnWindow);
DECLARE_int64(leftWindowSize);
DECLARE_int64(rightWindowSize);
DECLARE_int64(maxsil);
DECLARE_int64(minsil);
DECLARE_double(maxrate);
DECLARE_double(minrate);
DECLARE_int64(softwoffset);
DECLARE_double(softwrate);
DECLARE_double(softwstd);
DECLARE_bool(trainWithWindow);

/* ========== DISTRIBUTED TRAINING ========== */
DECLARE_bool(enable_distributed);
DECLARE_int64(world_rank);
DECLARE_int64(world_size);
DECLARE_string(rndv_filepath);

/* ========== FB SPECIFIC ========== */
DECLARE_string(target);
DECLARE_bool(everstoredb);
DECLARE_string(targettype);

} // namespace w2l
