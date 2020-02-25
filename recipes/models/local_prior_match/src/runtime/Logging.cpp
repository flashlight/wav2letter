/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/runtime/Logging.h"

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "recipes/models/local_prior_match/src/runtime/Defines.h"
#include "runtime/Serial.h"

namespace w2l {

LogHelper::LogHelper(
    int runIdx,
    std::string runPath,
    bool isMaster,
    bool logOnEpoch)
    : runIdx_(runIdx),
      runPath_(runPath),
      isMaster_(isMaster),
      logOnEpoch_(logOnEpoch) {
  if (isMaster_) {
    logFileName_ = getRunFile("log", runIdx_, runPath_);
    perfFileName_ = getRunFile("perf", runIdx_, runPath_);
    dirCreate(runPath_);
    std::ofstream logFile, perfFile;
    logFile.open(logFileName_);
    if (!logFile.is_open()) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    perfFile.open(perfFileName_);
    if (!perfFile.is_open()) {
      LOG(FATAL) << "failed to open perf file for writing";
    }
  }
}

void LogHelper::saveConfig(
    const std::unordered_map<std::string, std::string>& config) {
  if (!isMaster_) {
    return;
  }

  std::ofstream configFile(getRunFile("config", runIdx_, runPath_));
  cereal::JSONOutputArchive ar(configFile);
  ar(CEREAL_NVP(config));
}

void LogHelper::writeHeader(SSLTrainMeters& meters) {
  if (!isMaster_) {
    return;
  }

  std::ofstream perfFile;
  perfFile.open(perfFileName_);
  auto perfMsg = formatStatus(meters, 0, {}, false, true, "\t", true);
  appendToLog(perfFile, "# " + perfMsg);
}

void LogHelper::logStatus(
    SSLTrainMeters& mtrs,
    int64_t epoch,
    const std::unordered_map<std::string, double>& logFields) {
  syncMeter(mtrs);

  if (!isMaster_) {
    return;
  }

  try {
    std::ofstream logFile, perfFile;
    logFile.open(logFileName_, std::ofstream::out | std::ofstream::app);
    perfFile.open(perfFileName_, std::ofstream::out | std::ofstream::app);
    auto logMsg =
        formatStatus(mtrs, epoch, logFields, true, false, " | ", false);
    auto perfMsg =
        formatStatus(mtrs, epoch, logFields, false, true, " ", false);
    LOG_MASTER(INFO) << logMsg;
    appendToLog(logFile, logMsg);
    appendToLog(perfFile, perfMsg);
  } catch (const std::exception& ex) {
    LOG(ERROR) << "Error while writing logs: " << ex.what();
  }
}

std::string LogHelper::saveModel(
    const std::string& filename,
    const std::unordered_map<std::string, std::string>& config,
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<fl::FirstOrderOptimizer> netoptim /* = nullptr */,
    bool workerSave /* = false */) {
  if (!workerSave && !isMaster_) {
    return "";
  }

  std::string outputfile = getRunFile(filename, runIdx_, runPath_);
  try {
    if (netoptim) {
      W2lSerializer::save(outputfile, config, network, criterion, netoptim);
    } else {
      W2lSerializer::save(outputfile, config, network, criterion);
    }
  } catch (const std::exception& ex) {
    LOG(FATAL) << "Error while saving models to " + outputfile + ": "
               << ex.what();
  }

  return outputfile;
}

void LogHelper::logAndSaveModel(
    SSLTrainMeters& meters,
    const std::unordered_map<std::string, std::string>& config,
    std::shared_ptr<fl::Module> network,
    std::shared_ptr<SequenceCriterion> criterion,
    std::shared_ptr<fl::FirstOrderOptimizer> netoptim,
    const std::unordered_map<std::string, double>& logFields) {
  int iter = logOnEpoch_ ? std::stoi(config.at(kEpoch))
                         : std::stoi(config.at(kIteration));
  std::string tag = "last";
  if (FLAGS_itersave) {
    tag = logOnEpoch_ ? format("epoch_%04d", iter) : format("iter_%08d", iter);
  }

  logStatus(meters, iter, logFields);
  saveModel("model_" + tag + ".bin", config, network, criterion, netoptim);

  for (auto& s : meters.valid) {
    double verr = s.second.edits[kTarget].value()[0];
    auto sit = validminerrs_.find(s.first);
    if (sit == validminerrs_.end() || sit->second > verr) {
      validminerrs_[s.first] = verr;
      saveModel(
          "model_" + s.first + ".bin", config, network, criterion, netoptim);
    }
  }
}

std::string LogHelper::formatStatus(
    SSLTrainMeters& meters,
    int64_t epoch,
    const std::unordered_map<std::string, double>& logFields,
    bool verbose /* = false */,
    bool date /* = false */,
    const std::string& separator /* = " " */,
    bool headerOnly /* = false */) {
  std::string header, status;

  auto insertItem = [&](std::string key, std::string val) {
    if (verbose) {
      val = key + ": " + val;
    }
    header = header + (header.empty() ? "" : separator) + key;
    status = status + (status.empty() ? "" : separator) + val;
  };

  auto insertSSLDatasetMeters = [&insertItem](
                                    SSLDatasetMeters& meter, std::string tag) {
    for (auto& m : meter.values) {
      insertItem(tag + "-" + m.first, format("%10.5f", m.second.value()[0]));
    }
    for (auto& m : meter.edits) {
      insertItem(
          tag + "-" + m.first + "ER", format("%5.2f", m.second.value()[0]));
    }
  };

  if (date) {
    insertItem("date", format("%s", getCurrentDate().c_str()));
    insertItem("time", format("%s", getCurrentTime().c_str()));
  }

  if (logOnEpoch_) {
    insertItem("epoch", format("%8d", epoch));
  } else {
    insertItem("iter", format("%8d", epoch));
  }

  insertItem("lr", headerOnly ? "" : format("%4.6lf", logFields.at("lr")));

  int rt = meters.timer[kRuntime].value();
  insertItem(
      kRuntime,
      format("%02d:%02d:%02d", (rt / 60 / 60), (rt / 60) % 60, rt % 60));

  for (auto& m : meters.timer) {
    if (m.first == kRuntime) {
      continue;
    }
    insertItem(m.first + "(ms)", format("%.2f", m.second.value() * 1000));
  }

  for (auto& m : meters.values) {
    insertItem("train-" + m.first, format("%10.5f", m.second.value()[0]));
  }

  insertSSLDatasetMeters(meters.train, "train");
  for (auto& v : meters.valid) {
    insertSSLDatasetMeters(v.second, v.first);
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
    audioProcSec = audioProcSec * FLAGS_framestridems / 1000.0;
  } else {
    audioProcSec /= FLAGS_samplerate;
  }
  auto worldSize = fl::getWorldSize();
  double timeTakenSec = meters.timer[kTimer].value() * numsamples / worldSize;

  insertItem("hrs", format("%7.2f", audioProcSec / 3600.0));
  insertItem(
      "thrpt(sec/sec)",
      timeTakenSec > 0.0 ? format("%.2f", audioProcSec / timeTakenSec) : "n/a");
  return headerOnly ? header : status;
}

template <>
void syncMeter<SSLTrainMeters>(SSLTrainMeters& meters) {
  syncMeter(meters.stats);
  for (auto& m : meters.timer) {
    syncMeter(m.second);
  }
  for (auto& m : meters.values) {
    syncMeter(m.second);
  }
  syncMeter(meters.train);
  for (auto& m : meters.valid) {
    syncMeter(m.second);
  }
}

template <>
void syncMeter<SSLDatasetMeters>(SSLDatasetMeters& meters) {
  for (auto& m : meters.edits) {
    syncMeter(m.second);
  }
  for (auto& m : meters.values) {
    syncMeter(m.second);
  }
}

void resetTrainMeters(SSLTrainMeters& meters) {
  for (auto& m : meters.timer) {
    m.second.reset();
  }
  for (auto& m : meters.values) {
    m.second.reset();
  }
  meters.stats.reset();
  resetDatasetMeters(meters.train);
}

void stopTimeMeters(SSLTrainMeters& meters) {
  for (auto& m : meters.timer) {
    m.second.stop();
  }
}

void resetDatasetMeters(SSLDatasetMeters& meters) {
  for (auto& m : meters.edits) {
    m.second.reset();
  }
  for (auto& m : meters.values) {
    m.second.reset();
  }
}

double avgValidErr(SSLTrainMeters& meters) {
  double err = 0.0;
  for (auto& s : meters.valid) {
    err += s.second.edits[kTarget].value()[0];
  }
  err /= meters.valid.size();
  return err;
}

} // namespace w2l
