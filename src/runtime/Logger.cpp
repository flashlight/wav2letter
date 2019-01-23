/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Logger.h"

#include <thread>

#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Utils.h"

namespace w2l {
std::pair<std::string, std::string> getStatus(
    TrainMeters& meters,
    int64_t epoch,
    double lr,
    double lrcrit,
    bool verbose /* = false */,
    bool date /* = false */,
    const std::string& separator /* = " " */) {
  std::string errtype = "XER";
  errtype[0] = std::toupper(FLAGS_target[0]);
  std::string header, status;
  auto insertItem = [&](std::string key, std::string val) {
    if (verbose) {
      val = key + ": " + val;
    }
    header = header + (header.empty() ? "" : separator) + key;
    status = status + (status.empty() ? "" : separator) + val;
  };
  if (date) {
    insertItem("date", format("%s", getCurrentDate().c_str()));
    insertItem("time", format("%s", getCurrentTime().c_str()));
  }
  insertItem("epoch", format("%8d", epoch));
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
  insertItem("loss", format("%10.5f", meters.loss.value()[0]));

  insertItem("train-" + errtype, format("%5.2f", meters.train.edit.value()[0]));
  for (auto& v : meters.valid) {
    insertItem(
        v.first + "-" + errtype, format("%5.2f", v.second.edit.value()[0]));
  }
  auto stats = meters.stats.value();
  auto numsamples = std::max<int64_t>(stats[4], 1);
  auto isztotal = stats[0];
  auto tsztotal = stats[1];
  auto tszmax = stats[3];
  insertItem("avg-isz", format("%03d", isztotal / numsamples));
  insertItem("avg-tsz", format("%03d", tsztotal / numsamples));
  insertItem("max-tsz", format("%03d", tszmax));

  double audioProcSec = isztotal * FLAGS_batchsize;
  if (FLAGS_pow || FLAGS_mfcc || FLAGS_mfsc) {
    audioProcSec = audioProcSec * kFrameStrideMs / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = meters.timer.value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");
  return {header, status};
}

void print2file(std::ofstream& fs, const std::string& logstr) {
  auto attempsLeft = kTryCatchAttempts;
  do {
    try {
      fs << logstr << std::endl;
      break;
    } catch (...) {
      --attempsLeft;
      LOG_IF(ERROR, attempsLeft <= 0) << "Error while logging to file !";
      std::this_thread::sleep_for(std::chrono::seconds(kTryCatchWaitSec));
    }
  } while (attempsLeft > 0);
}

af::array allreduceGet(fl::AverageValueMeter& mtr) {
  auto mtrVal = mtr.value();
  mtrVal[0] *= mtrVal[2];
  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::EditDistanceMeter& mtr) {
  auto mtrVal = mtr.value();
  mtrVal[0] = mtrVal[0] * mtrVal[1] / 100;
  mtrVal[2] = mtrVal[2] * mtrVal[1] / 100;
  mtrVal[3] = mtrVal[3] * mtrVal[1] / 100;
  mtrVal[4] = mtrVal[4] * mtrVal[1] / 100;

  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(SpeechStatMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  // Caveat: maxInputSz_, maxTargetSz_ would be approximate
  mtrVal[2] *= mtrVal[4];
  mtrVal[3] *= mtrVal[4];
  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::CountMeter& mtr) {
  auto mtrVal0 = mtr.value();
  std::vector<long long> mtrVal(mtrVal0.begin(), mtrVal0.end());
  return af::array(mtrVal.size(), mtrVal.data());
}

af::array allreduceGet(fl::TimeMeter& mtr) {
  return af::constant(mtr.value(), 1, af::dtype::f64);
}

void allreduceSet(fl::AverageValueMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<double>(val);
  if (valVec[2] != 0) {
    valVec[0] /= valVec[2];
  }
  mtr.add(valVec[0], valVec[2]);
}

void allreduceSet(fl::EditDistanceMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<double>(val);
  mtr.add(
      static_cast<int64_t>(valVec[1]),
      static_cast<int64_t>(valVec[2]),
      static_cast<int64_t>(valVec[3]),
      static_cast<int64_t>(valVec[4]));
}

void allreduceSet(SpeechStatMeter& mtr, af::array& val) {
  mtr.reset();
  // Caveat: maxInputSz_, maxTargetSz_ would be approximate
  auto valVec = afToVector<int64_t>(val);
  SpeechStats stats;
  auto denom = (valVec[4] == 0) ? 1 : valVec[4];
  stats.totalInputSz_ = valVec[0];
  stats.totalTargetSz_ = valVec[1];
  stats.maxInputSz_ = valVec[2] / denom;
  stats.maxTargetSz_ = valVec[3] / denom;
  stats.numSamples_ = valVec[4];
  mtr.add(stats);
}

void allreduceSet(fl::CountMeter& mtr, af::array& val) {
  mtr.reset();
  auto valVec = afToVector<long long>(val);
  for (size_t i = 0; i < valVec.size(); ++i) {
    mtr.add(i, valVec[i]);
  }
}

void allreduceSet(fl::TimeMeter& mtr, af::array& val) {
  auto worldSize = fl::getWorldSize();
  auto valVec = afToVector<double>(val);
  mtr.set(valVec[0] / worldSize);
}

template <>
void syncMeter<TrainMeters>(TrainMeters& mtrs) {
  syncMeter(mtrs.loss);
  syncMeter(mtrs.stats);
  syncMeter(mtrs.runtime);
  syncMeter(mtrs.timer);
  syncMeter(mtrs.fwdtimer);
  syncMeter(mtrs.critfwdtimer);
  syncMeter(mtrs.bwdtimer);
  syncMeter(mtrs.optimtimer);
  syncMeter(mtrs.train.edit);
  syncMeter(mtrs.train.wordedit);
  for (auto& v : mtrs.valid) {
    syncMeter(v.second.edit);
    syncMeter(v.second.wordedit);
  }
}

} // namespace w2l
