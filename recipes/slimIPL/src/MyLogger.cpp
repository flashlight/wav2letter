/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "MyLogger.h"

#include <thread>

#include "flashlight/lib/common/String.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/common/Flags.h"

using fl::lib::format;
using fl::lib::getCurrentDate;
using fl::lib::getCurrentTime;
using fl::lib::retryWithBackoff;
using fl::pkg::runtime::afToVector;
using namespace fl::pkg::speech;

namespace slimIPL {

std::string getLogString(
    TrainMetersMy& meters,
    const std::unordered_map<std::string, double>& validDecoderWer,
    int64_t epoch,
    int64_t nupdates,
    double lr,
    double lrcrit,
    const std::string& separator /* = " | " */) {
  std::string status;
  auto insertItem = [&](std::string key, std::string val) {
    val = key + ": " + val;
    status = status + (status.empty() ? "" : separator) + val;
  };
  insertItem("epoch", format("%8d", epoch));
  insertItem("nupdates", format("%12d", nupdates));
  insertItem("lr", format("%4.6lf", lr));
  insertItem("lrcriterion", format("%4.6lf", lrcrit));

  int rt = meters.runtime.value();
  insertItem(
      "runtime",
      format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60));
  insertItem("bch(ms)", format("%.2f", meters.timer.value() * 1000));
  insertItem("smp(ms)", format("%.2f", meters.sampletimer.value() * 1000));
  insertItem("fwd(ms)", format("%.2f", meters.fwdtimer.value() * 1000));
  insertItem(
      "crit-fwd(ms)", format("%.2f", meters.critfwdtimer.value() * 1000));
  insertItem("bwd(ms)", format("%.2f", meters.bwdtimer.value() * 1000));
  insertItem("optim(ms)", format("%.2f", meters.optimtimer.value() * 1000));
  insertItem("loss", format("%10.5f", meters.train.loss.value()[0]));

  insertItem("train-TER", format("%5.2f", meters.train.tknEdit.errorRate()[0]));
  insertItem("train-WER", format("%5.2f", meters.train.wrdEdit.errorRate()[0]));
  for (auto& v : meters.valid) {
    insertItem(v.first + "-loss", format("%10.5f", v.second.loss.value()[0]));
    insertItem(
        v.first + "-TER", format("%5.2f", v.second.tknEdit.errorRate()[0]));
    insertItem(
        v.first + "-WER", format("%5.2f", v.second.wrdEdit.errorRate()[0]));
    auto vDecoderIter = validDecoderWer.find(v.first);
    if (vDecoderIter != validDecoderWer.end()) {
      insertItem(
          v.first + "-WER-decoded", format("%5.2f", vDecoderIter->second));
    }
  }
  auto stats = meters.stats.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto numbatches = std::max<int64_t>(stats[5], 1);
  // assumed to be in ms of original audios
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  auto iszAvrFrames = isztotal / numsamples;
  if (FLAGS_features_type != fl::pkg::speech::kFeaturesRaw) {
    iszAvrFrames = iszAvrFrames / FLAGS_framestridems;
  } else {
    iszAvrFrames = iszAvrFrames / 1000 * FLAGS_samplerate;
  }
  insertItem("avg-isz", format("%03d", iszAvrFrames));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  auto worldSize = fl::getWorldSize();
  double timeTakenSec = meters.timer.value() * numbatches / worldSize;

  insertItem("avr-batchsz", format("%7.2f", float(numsamples) / numbatches));
  insertItem("hrs", format("%7.2f", isztotal / 1000 / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", isztotal / 1000 / timeTakenSec)
                         : "n/a");
  insertItem("timestamp", getCurrentDate() + " " + getCurrentTime());
  insertItem("loss Unsup", format("%10.5f", meters.trainUnsup.loss.value()[0]));

  insertItem(
      "train-TER Unsup",
      format("%5.2f", meters.trainUnsup.tknEdit.errorRate()[0]));
  insertItem(
      "train-WER Unsup",
      format("%5.2f", meters.trainUnsup.wrdEdit.errorRate()[0]));

  return status;
}

void syncMeter(TrainMetersMy& mtrs) {
  fl::pkg::runtime::syncMeter(mtrs.stats);
  fl::pkg::runtime::syncMeter(mtrs.runtime);
  fl::pkg::runtime::syncMeter(mtrs.timer);
  fl::pkg::runtime::syncMeter(mtrs.fwdtimer);
  fl::pkg::runtime::syncMeter(mtrs.critfwdtimer);
  fl::pkg::runtime::syncMeter(mtrs.bwdtimer);
  fl::pkg::runtime::syncMeter(mtrs.optimtimer);
  fl::pkg::runtime::syncMeter(mtrs.train.tknEdit);
  fl::pkg::runtime::syncMeter(mtrs.train.wrdEdit);
  fl::pkg::runtime::syncMeter(mtrs.train.loss);
  fl::pkg::runtime::syncMeter(mtrs.trainUnsup.tknEdit);
  fl::pkg::runtime::syncMeter(mtrs.trainUnsup.wrdEdit);
  fl::pkg::runtime::syncMeter(mtrs.trainUnsup.loss);
  for (auto& v : mtrs.valid) {
    fl::pkg::runtime::syncMeter(v.second.tknEdit);
    fl::pkg::runtime::syncMeter(v.second.wrdEdit);
    fl::pkg::runtime::syncMeter(v.second.loss);
  }
}

} // namespace slimIPL
