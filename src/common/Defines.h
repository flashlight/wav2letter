/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include "libraries/common/Defines.h"

#include <memory>
#include <unordered_map>

#include <gflags/gflags.h>

#define W2L_VERSION "0.1"

namespace w2l {

namespace detail {

using DeprecatedFlagsMap = std::unordered_map<std::string, std::string>;

/**
 * Creates and maintains a map of deprecated flags. The map takes
 * a deprecated flag name to a new flag name; for instance, the entry:
 * ---> {myOldFlag, myNewFlag}
 * corresponds to the deprecation of myOldFlag
 */
DeprecatedFlagsMap& getDeprecatedFlags();

// Adds a flag to the global deprecated map
void addDeprecatedFlag(
    const std::string& depreactedFlagName,
    const std::string& newFlagName);

// Whether the flag has been explicitly set either from the cmd line or
// de-serialization
bool isFlagSet(const std::string& name);

} // namespace detail

/**
 * Globally-accessible and recommended to be called immediately after gflags
 * have been parsed and initialized. Does a few things:
 * - Sets the value of the new flag to be the value of the old flag
 * - Displays a message indicating that the old flag is deprecated and the new
 * flag shoule be used.
 *
 * Behavior is as follows:
 * - Throws if the user set both the deprecated flag and the new flag.
 * - Sets the new flag equal to the deprecated flag if the user only set the
 * deprecated flag.
 * - Does nothing if the user set neither the new nor the deprecated flag, or if
 * the user correctly set only the new flag and not the deprecated flag.
 */
void handleDeprecatedFlags();

/**
 * Deprecate a command line flag.
 *
 * USAGE:
 *   DEPRECATE_FLAGS(myOldFlagName, my_new_flag_name)
 */
#define DEPRECATE_FLAGS(DEPRECATED, NEW) \
  w2l::detail::addDeprecatedFlag(#DEPRECATED, #NEW);

// Dataset indices
// If a new field is added, `kNumDataIdx` should be modified accordingly.
constexpr size_t kInputIdx = 0;
constexpr size_t kTargetIdx = 1;
constexpr size_t kWordIdx = 2;
constexpr size_t kSampleIdx = 3;
constexpr size_t kNumDataIdx = 4; // total number of dataset indices

// Various constants used in w2l
constexpr const char* kTrainMode = "train";
constexpr const char* kContinueMode = "continue";
constexpr const char* kForkMode = "fork";
constexpr const char* kGflags = "gflags";
constexpr const char* kCommandLine = "commandline";
constexpr const char* kUserName = "username";
constexpr const char* kHostName = "hostname";
constexpr const char* kTimestamp = "timestamp";
constexpr const char* kRunIdx = "runIdx";
constexpr const char* kRunPath = "runPath";
constexpr const char* kProgramName = "programname";
constexpr const char* kEpoch = "epoch";
constexpr const char* kUpdates = "updates";
constexpr const char* kSGDOptimizer = "sgd";
constexpr const char* kAdamOptimizer = "adam";
constexpr const char* kRMSPropOptimizer = "rmsprop";
constexpr const char* kAdadeltaOptimizer = "adadelta";
constexpr const char* kAdagradOptimizer = "adagrad";
constexpr const char* kAMSgradOptimizer = "amsgrad";
constexpr const char* kNovogradOptimizer = "novograd";
constexpr const char* kCtcCriterion = "ctc";
constexpr const char* kAsgCriterion = "asg";
constexpr const char* kSeq2SeqCriterion = "seq2seq";
constexpr const char* kTransformerCriterion = "transformer";
constexpr int kTargetPadValue = -1;

// Feature params
constexpr int kLifterParam = 22;
constexpr int kPrefetchSize = 2;

/* ========== DATA OPTIONS ========== */

DECLARE_string(train);
DECLARE_string(valid);
DECLARE_string(test);
DECLARE_int64(batchsize);
DECLARE_int64(validbatchsize);
DECLARE_string(input);
DECLARE_int64(samplerate);
DECLARE_int64(channels);
DECLARE_string(tokens);
DECLARE_bool(usewordpiece);
DECLARE_int64(replabel);
DECLARE_string(surround);
DECLARE_bool(noresample);
DECLARE_bool(eostoken);
DECLARE_string(dataorder);
DECLARE_int64(inputbinsize);
DECLARE_int64(outputbinsize);
DECLARE_bool(blobdata);
DECLARE_string(wordseparator);
DECLARE_double(sampletarget);

/* ========== FILTERING OPTIONS ========== */

DECLARE_int64(minisz);
DECLARE_int64(maxisz);
DECLARE_int64(mintsz);
DECLARE_int64(maxtsz);

/* ========== NORMALIZATION OPTIONS ========== */

DECLARE_int64(localnrmlleftctx);
DECLARE_int64(localnrmlrightctx);
DECLARE_string(onorm);
DECLARE_bool(sqnorm);
DECLARE_bool(lrcosine);

/* ========== LEARNING HYPER-PARAMETER OPTIONS ========== */

DECLARE_int64(iter);
DECLARE_bool(itersave);
DECLARE_double(lr);
DECLARE_double(momentum);
DECLARE_double(weightdecay);
DECLARE_bool(sqnorm);
DECLARE_double(lrcrit);
DECLARE_int64(warmup);
DECLARE_int64(saug_start_update);
DECLARE_int64(lr_decay);
DECLARE_int64(lr_decay_step);
DECLARE_double(maxgradnorm);
DECLARE_double(adambeta1); // TODO rename into optim beta1
DECLARE_double(adambeta2); // TODO rename into optim beta2
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
DECLARE_int64(framesizems);
DECLARE_int64(framestridems);

/* ========== SPECAUGMENT OPTIONS ========== */

DECLARE_int64(saug_fmaskf);
DECLARE_int64(saug_fmaskn);
DECLARE_int64(saug_tmaskt);
DECLARE_double(saug_tmaskp);
DECLARE_int64(saug_tmaskn);

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
DECLARE_double(pcttraineval);

/* ========== ARCHITECTURE OPTIONS ========== */

DECLARE_string(arch);
DECLARE_string(criterion);
DECLARE_int64(encoderdim);

// Seq2Seq Transformer decoder
DECLARE_int64(am_decoder_tr_layers);
DECLARE_double(am_decoder_tr_dropout);
DECLARE_double(am_decoder_tr_layerdrop);

/* ========== DECODER OPTIONS ========== */

DECLARE_bool(show);
DECLARE_bool(showletters);
DECLARE_bool(logadd);
DECLARE_bool(uselexicon);
DECLARE_bool(isbeamdump);

DECLARE_string(smearing);
DECLARE_string(lmtype);
DECLARE_string(lexicon);
DECLARE_string(lm_vocab);
DECLARE_string(emission_dir);
DECLARE_string(lm);
DECLARE_string(am);
DECLARE_string(sclite);
DECLARE_string(decodertype);

DECLARE_double(lmweight);
DECLARE_double(wordscore);
DECLARE_double(silscore);
DECLARE_double(unkscore);
DECLARE_double(eosscore);
DECLARE_double(beamthreshold);

DECLARE_int32(maxload);
DECLARE_int32(maxword);
DECLARE_int32(beamsize);
DECLARE_int32(beamsizetoken);
DECLARE_int32(nthread_decoder_am_forward);
DECLARE_int32(nthread_decoder);
DECLARE_int32(lm_memory);

DECLARE_int32(emission_queue_size);

// Seq2Seq
DECLARE_double(smoothingtemperature);
DECLARE_int32(attentionthreshold);

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
DECLARE_int64(attndim);
DECLARE_int64(attnconvchannel);
DECLARE_int64(attnconvkernel);
DECLARE_int64(numattnhead);
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
DECLARE_int64(pretrainWindow);
DECLARE_double(gumbeltemperature);
DECLARE_int64(decoderrnnlayer);
DECLARE_int64(decoderattnround);
DECLARE_double(decoderdropout);

/* ========== DISTRIBUTED TRAINING ========== */
DECLARE_bool(enable_distributed);
DECLARE_int64(world_rank);
DECLARE_int64(world_size);
DECLARE_int64(max_devices_per_node);
DECLARE_string(rndv_filepath);

/* ========== FB SPECIFIC ========== */
DECLARE_string(target);
DECLARE_bool(everstoredb);
DECLARE_bool(use_memcache);

} // namespace w2l
