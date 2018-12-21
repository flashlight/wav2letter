/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <regex>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "data/W2lDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "module/module.h"
#include "runtime/runtime.h"

#ifdef BUILD_FB_DEPENDENCIES
#include "fb/W2lEverstoreDataset.h"
#endif

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + std::string() +
      " continue [directory] [flags]\n or " + std::string(argv[0]) +
      " fork [directory/model] [flags]");

  /* ===================== Parse Options ===================== */
  int runidx = 1; // current #runs in this path
  std::string runpath; // current experiment path
  std::string reloadpath; // path to model to reload
  int startEpoch = 0;
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  if (strcmp(argv[1], "train") == 0) {
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    runpath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else if (strcmp(argv[1], "continue") == 0) {
    runpath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runidx, runpath))) {
      ++runidx;
    }
    reloadpath = getRunFile("model_last.bin", runidx - 1, runpath);
    LOG(INFO) << "reload path is " << reloadpath;
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadpath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadpath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadpath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    if (argc > 3) {
      LOG(INFO) << "Parsing command line flags";
      LOG(INFO) << "Overriding flags should be mutable when using `continue`";
      gflags::ParseCommandLineFlags(&argc, &argv, false);
    }
    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    auto epoch = cfg.find(kEpoch);
    if (epoch == cfg.end()) {
      LOG(WARNING) << "Did not find epoch to start from, starting from 0.";
    } else {
      startEpoch = std::stoi(epoch->second);
    }
  } else if (strcmp(argv[1], "fork") == 0) {
    reloadpath = argv[2];
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadpath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadpath;
    }

    LOG(INFO) << "Reading flags from config file " << reloadpath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    if (argc > 3) {
      LOG(INFO) << "Parsing command line flags";
      LOG(INFO) << "Overriding flags should be mutable when using `fork`";
      gflags::ParseCommandLineFlags(&argc, &argv, false);
    }

    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file" << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    runpath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  af::setMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);

  maybeInitDistributedEnv(
      FLAGS_enable_distributed,
      FLAGS_world_rank,
      FLAGS_world_size,
      FLAGS_rndv_filepath);

  auto worldRank = fl::getWorldRank();
  auto worldSize = fl::getWorldSize();

  bool isMaster = (worldRank == 0);

  LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  LOG_MASTER(INFO) << "Experiment path: " << runpath;
  LOG_MASTER(INFO) << "Experiment runidx: " << runidx;

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, getEnvVar("USER")},
      {kHostName, getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runidx)},
      {kRunPath, runpath}};

  auto validsets = split(',', trim(FLAGS_valid));

  /* ===================== Create Dictionary ===================== */
  Dictionary dict = makeDictionary();
  int numClasses = dict.indexSize();
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::SequenceCriterion> criterion;
  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  if (reloadpath.empty()) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();
    // Encoder network, works on audio
    network = createW2lSeqModule(archfile, numFeatures, numClasses);

    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<fl::CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<fl::ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqCriterion) {
      criterion = std::make_shared<fl::Seq2SeqCriterion>(
          buildSeq2Seq(numClasses, dict.getIndex(kEosToken)));
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }
  } else {
    std::unordered_map<std::string, std::string> cfg; // unused
    W2lSerializer::load(reloadpath, cfg, network, criterion);
  }
  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();

  std::shared_ptr<fl::LinSegCriterion> linseg;
  if (FLAGS_linseg > 0) {
    if (FLAGS_criterion != kAsgCriterion) {
      LOG(FATAL) << "linseg may only be used with ASG criterion";
    }
    linseg = std::make_shared<fl::LinSegCriterion>(numClasses, scalemode);
    linseg->setParams(criterion->param(0), 0);
    LOG_MASTER(INFO) << "[Criterion] " << linseg->prettyString()
                     << " (for first " << FLAGS_linseg << " epochs)";
  }

  /* ===================== Meters ===================== */
  TrainMeters meters;
  for (const auto& s : validsets) {
    meters.valid[s] = EditDistMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validsets) {
    validminerrs[s] = DBL_MAX;
  }

  /* ===================== Logging ===================== */
  std::ofstream perfile, logfile;
  if (isMaster) {
    dirCreate(runpath);
    perfile.open(getRunFile("perf", runidx, runpath), std::ios::out);
    logfile.open(getRunFile("log", runidx, runpath), std::ios::out);

    // write header to perfile
    auto msgp = getStatus(meters, 0, 0, 0, false, true, "\t").first;
    print2file(perfile, "# " + msgp);
    // write config
    std::ofstream cfgfile(getRunFile("config", runidx, runpath));
    cereal::JSONOutputArchive ar(cfgfile);
    ar(CEREAL_NVP(config));
  }

  auto logStatus =
      [&perfile, &logfile, isMaster](
          TrainMeters& mtrs, int64_t epoch, double lr, double lrcrit) {
        syncMeter(mtrs);

        if (isMaster) {
          auto msgl =
              getStatus(mtrs, epoch, lr, lrcrit, true, false, " | ").second;
          auto msgp = getStatus(mtrs, epoch, lr, lrcrit, false, true).second;
          LOG_MASTER(INFO) << msgl;
          print2file(logfile, msgl);
          print2file(perfile, msgp);
        }
      };

  auto saveModels = [&](int iter) {
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);

      std::string filename;
      if (FLAGS_itersave) {
        filename =
            getRunFile(format("model_iter_%03d.bin", iter), runidx, runpath);
      } else {
        filename = getRunFile("model_last.bin", runidx, runpath);
      }
      // save last model
      W2lSerializer::save(filename, config, network, criterion);

      // save if better than ever for one valid
      for (const auto& v : validminerrs) {
        double verr = meters.valid[v.first].edit.value()[0];
        if (verr < validminerrs[v.first]) {
          validminerrs[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname =
              getRunFile("model_" + cleaned_v + ".bin", runidx, runpath);
          W2lSerializer::save(vfname, config, network, criterion);
        }
      }
    }
  };

  /* ===================== Create Dataset ===================== */
  auto createDataset = [worldRank, worldSize, &dicts](const std::string& path) {
    std::shared_ptr<W2lDataset> ds;
    if (FLAGS_everstoredb) {
#ifdef BUILD_FB_DEPENDENCIES
      W2lEverstoreDataset::init(); // Required for everstore client
      ds = std::make_shared<W2lEverstoreDataset>(
          path, dicts, FLAGS_batchsize, worldRank, worldSize, FLAGS_targettype);
#else
      LOG(FATAL) << "W2lEverstoreDataset not supported: "
                 << "build with -DBUILD_FB_DEPENDENCIES";
#endif
    } else {
      ds = std::make_shared<W2lNumberedFilesDataset>(
          path, dicts, FLAGS_batchsize, worldRank, worldSize, FLAGS_datadir);
    }
    return ds;
  };

  auto trainds = createDataset(FLAGS_train);

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    trainds->shuffle(FLAGS_seed);
  }

  std::unordered_map<std::string, std::shared_ptr<W2lDataset>> validds;
  for (const auto& s : validsets) {
    validds[s] = createDataset(s);
  }

  /* ===================== Hooks ===================== */
  auto evalOutput = [&dicts, &criterion](
                        const af::array& op,
                        const af::array& target,
                        fl::EditDistanceMeter& mtr) {
    auto batchsz = op.dims(2);
    for (int b = 0; b < batchsz; ++b) {
      auto tgt = target(af::span, b);
      auto viterbipath =
          afToVector<int>(criterion->viterbiPath(op(af::span, af::span, b)));
      auto tgtraw = afToVector<int>(tgt);

      // Remove `-1`s appended to the target for batching (if any)
      auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(labellen);

      // remap actual, predicted targets for evaluating edit distance error
      if (dicts.find(kTargetIdx) == dicts.end()) {
        LOG(FATAL) << "Dictionary not provided for target: " << kTargetIdx;
      }
      auto tgtDict = dicts.find(kTargetIdx)->second;

      if (FLAGS_criterion == kCtcCriterion ||
          FLAGS_criterion == kAsgCriterion) {
        uniq(viterbipath);
      }
      if (FLAGS_criterion == kCtcCriterion) {
        auto blankidx = tgtDict.getIndex(kBlankToken);
        viterbipath.erase(
            std::remove(viterbipath.begin(), viterbipath.end(), blankidx),
            viterbipath.end());
      }

      remapLabels(viterbipath, tgtDict);
      remapLabels(tgtraw, tgtDict);

      mtr.add(
          viterbipath.data(), tgtraw.data(), viterbipath.size(), tgtraw.size());
    }
  };

  auto test = [&evalOutput](
                  std::shared_ptr<fl::Module> ntwrk,
                  std::shared_ptr<fl::Loss> crit,
                  std::shared_ptr<W2lDataset> testds,
                  EditDistMeters& mtrs) {
    ntwrk->eval();
    crit->eval();
    mtrs.edit.reset();
    mtrs.wordedit.reset();

    for (auto& sample : *testds) {
      auto output = ntwrk->forward(fl::input(sample[kInputIdx]));
      evalOutput(output.array(), sample[kTargetIdx], mtrs.edit);
    }
  };

  double gradNorm = 1.0 / (FLAGS_batchsize * worldSize);

  auto train = [&meters,
                &test,
                &logStatus,
                &saveModels,
                &evalOutput,
                &validds,
                gradNorm,
                startEpoch](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<fl::Loss> crit,
                   std::shared_ptr<W2lDataset> trainset,
                   fl::FirstOrderOptimizer& netopt,
                   fl::FirstOrderOptimizer& critopt,
                   bool clampCrit,
                   int nepochs) {
    fl::distributeModuleGrads(ntwrk, gradNorm);
    fl::distributeModuleGrads(crit, gradNorm);

    meters.loss.reset();
    meters.train.edit.reset();
    meters.train.wordedit.reset();

    fl::allReduceParameters(ntwrk);
    fl::allReduceParameters(crit);

    auto resetTimeStatMeters = [&meters]() {
      meters.runtime.reset();
      meters.stats.reset();
      meters.sampletimer.reset();
      meters.fwdtimer.reset();
      meters.critfwdtimer.reset();
      meters.bwdtimer.reset();
      meters.optimtimer.reset();
      meters.timer.reset();
    };
    auto runValAndSaveModel = [&](int64_t epoch, double lr, double lrcrit) {
      meters.runtime.stop();
      meters.timer.stop();
      meters.sampletimer.stop();
      meters.fwdtimer.stop();
      meters.critfwdtimer.stop();
      meters.bwdtimer.stop();
      meters.optimtimer.stop();

      // valid
      for (auto& vds : validds) {
        test(ntwrk, crit, vds.second, meters.valid[vds.first]);
      }

      // print status
      logStatus(meters, epoch, lr, lrcrit);
      // save last and best models
      saveModels(epoch);
      // reset meters for next readings
      meters.loss.reset();
      meters.train.edit.reset();
      meters.train.wordedit.reset();
    };

    int64_t curEpoch = startEpoch;
    int64_t sampleIdx = 0;
    while (curEpoch < nepochs) {
      ++curEpoch;
      ntwrk->train();
      crit->train();
      if (FLAGS_reportiters == 0) {
        resetTimeStatMeters();
      }
      if (!FLAGS_noresample) {
        LOG_MASTER(INFO) << "Shuffling trainset";
        trainset->shuffle(curEpoch /* seed */);
      }
      af::sync();
      meters.sampletimer.resume();
      meters.runtime.resume();
      meters.timer.resume();
      LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
      for (auto& sample : *trainset) {
        // meters
        ++sampleIdx;
        af::sync();
        meters.timer.incUnit();
        meters.sampletimer.stopAndIncUnit();
        meters.stats.add(sample[kInputIdx], sample[kTargetIdx]);
        if (af::anyTrue<bool>(af::isNaN(sample[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(sample[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values";
        }

        // forward
        meters.fwdtimer.resume();
        auto output = ntwrk->forward(fl::input(sample[kInputIdx]));
        af::sync();
        meters.critfwdtimer.resume();
        auto loss = crit->forward(output, fl::noGrad(sample[kTargetIdx]));
        af::sync();
        meters.fwdtimer.stopAndIncUnit();
        meters.critfwdtimer.stopAndIncUnit();

        if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
          LOG(FATAL) << "Loss has NaN values";
        }
        auto batchLoss = afToVector<float>(loss.array());
        for (const auto lossval : batchLoss) {
          meters.loss.add(lossval);
        }

        int64_t batchIdx = (sampleIdx - 1) % trainset->size();
        int64_t globalBatchIdx = trainset->getGlobalBatchIdx(batchIdx);
        if (globalBatchIdx % 100 < FLAGS_pcttraineval) {
          evalOutput(output.array(), sample[kTargetIdx], meters.train.edit);
        }

        // backward
        meters.bwdtimer.resume();
        netopt.zeroGrad();
        critopt.zeroGrad();
        loss.backward();

        af::sync();
        meters.bwdtimer.stopAndIncUnit();
        meters.optimtimer.resume();

        if (FLAGS_maxgradnorm > 0) {
          auto params = ntwrk->params();
          if (clampCrit) {
            auto critparams = crit->params();
            params.insert(params.end(), critparams.begin(), critparams.end());
          }
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }
        critopt.step();
        netopt.step();
        af::sync();
        meters.optimtimer.stopAndIncUnit();
        meters.sampletimer.resume();

        if (FLAGS_reportiters > 0 && sampleIdx % FLAGS_reportiters == 0) {
          runValAndSaveModel(curEpoch, netopt.getLr(), critopt.getLr());
          resetTimeStatMeters();
          ntwrk->train();
          crit->train();
          meters.sampletimer.resume();
          meters.runtime.resume();
          meters.timer.resume();
        }
      }
      af::sync();
      if (FLAGS_reportiters == 0) {
        runValAndSaveModel(curEpoch, netopt.getLr(), critopt.getLr());
      }
      double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
      netopt.setLr(lrScale * FLAGS_lr);
      critopt.setLr(lrScale * FLAGS_lrcrit);
    }
  };

  /* ===================== Train ===================== */
  if (FLAGS_linseg > 0) {
    auto linlr = FLAGS_linlr >= 0.0 ? FLAGS_linlr : FLAGS_lr;
    auto linlrcrit = FLAGS_linlrcrit >= 0.0 ? FLAGS_linlrcrit : FLAGS_lrcrit;

    auto linNetoptim = fl::SGDOptimizer(
        network->params(), linlr, FLAGS_momentum, FLAGS_weightdecay);
    auto linCritoptim = fl::SGDOptimizer(linseg->params(), linlrcrit, 0.0, 0.0);

    train(
        network,
        linseg,
        trainds,
        linNetoptim,
        linCritoptim,
        false /* clampCrit */,
        FLAGS_linseg);
  }

  // TODO - add support for other optimizers
  auto netoptim = fl::SGDOptimizer(
      network->params(), FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
  auto critoptim =
      fl::SGDOptimizer(criterion->params(), FLAGS_lrcrit, 0.0, 0.0);

  train(
      network,
      criterion,
      trainds,
      netoptim,
      critoptim,
      true /* clampCrit */,
      FLAGS_iter);

  LOG_MASTER(INFO) << "Finished training";

  perfile.close();
  logfile.close();

  return 0;
}
