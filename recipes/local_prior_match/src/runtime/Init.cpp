/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "recipes/models/local_prior_match/src/runtime/Init.h"

#include <tuple>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/FlashlightUtils.h"
#include "recipes/models/local_prior_match/src/runtime/Defines.h"
#include "recipes/models/local_prior_match/src/runtime/Utils.h"
#include "runtime/Serial.h"

namespace w2l {
std::unordered_map<std::string, std::string> setFlags(int argc, char** argv) {
  auto readNewFlags = [&]() {
    auto oldFlagsfile = FLAGS_flagsfile;
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (!FLAGS_flagsfile.empty() && FLAGS_flagsfile != oldFlagsfile) {
      LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
  };

  auto loadOldFlags = [&](const std::string& reloadPath) {
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadPath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    auto epoch = cfg.find(kEpoch);
    LOG_IF(WARNING, epoch == cfg.end())
        << "Did not find epoch to start from, starting from 0.";

    auto startEp = epoch == cfg.end() ? 0 : std::stoi(epoch->second);
    auto startIt =
        cfg.find(kIteration) == cfg.end() ? 0 : std::stoi(cfg[kIteration]);

    return std::make_pair(startEp, startIt);
  };

  std::string runStatus = argv[1];
  std::string runPath; // current experiment path
  int runIdx = 1; // current #runs in this path
  std::string reloadPath; // path to model to reload
  std::string propPath; // path to proposal model to reload
  int startEpoch = 0;
  int startIter = 0;

  if (runStatus == kTrainMode) {
    readNewFlags();
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
    propPath = FLAGS_proposalModel;
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    // this assumes that FLAGS_itersave wasn't set
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::tie(startEpoch, startIter) = loadOldFlags(reloadPath);
    readNewFlags();
    propPath = getRunFile("prop.bin", runIdx - 1, runPath);
  } else if (runStatus == kForkMode) {
    reloadPath = argv[2];
    loadOldFlags(reloadPath);
    readNewFlags();
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
    propPath = FLAGS_proposalModel;
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, argvs[0]},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      {kUserName, getEnvVar("USER")},
      {kHostName, getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath},
      // extra fields defined in localPriorMatchOss/runtime/Defines.h
      {kReloadPath, reloadPath},
      {kPropPath, propPath},
      {kRunStatus, runStatus},
      {kStartEpoch, std::to_string(startEpoch)},
      {kStartIter, std::to_string(startIter)}};

  return config;
}

} // namespace w2l
