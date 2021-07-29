/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <map>
#include "flashlight/app/asr/runtime/Logger.h"
#include "flashlight/app/asr/runtime/SpeechStatMeter.h"
#include "flashlight/fl/flashlight.h"

namespace slimIPL {

struct TrainMetersMy {
  fl::TimeMeter runtime;
  fl::TimeMeter timer{true};
  fl::TimeMeter sampletimer{true};
  fl::TimeMeter fwdtimer{true}; // includes network + criterion time
  fl::TimeMeter critfwdtimer{true};
  fl::TimeMeter bwdtimer{true}; // includes network + criterion time
  fl::TimeMeter optimtimer{true};

  fl::app::asr::DatasetMeters train;
  fl::app::asr::DatasetMeters trainUnsup;
  std::map<std::string, fl::app::asr::DatasetMeters> valid;

  fl::app::asr::SpeechStatMeter stats;
};

/*
 * Utility function to log results (learning rate, WER, TER, epoch, timing)
 * From gflags it uses FLAGS_batchsize, FLAGS_features_type
 * FLAGS_framestridems, FLAGS_samplerate
 */
std::string getLogString(
    TrainMetersMy& meters,
    const std::unordered_map<std::string, double>& dmErrs,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    const std::string& separator = " | ");

void syncMeter(TrainMetersMy& mtrs);
} // namespace slimIPL
