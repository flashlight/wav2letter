/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + exec +
      " continue [directory] [flags]\n or " + exec +
      " fork [directory/model] [flags]");

  /* ===================== Parse Options ===================== */
  int runIdx = 1; // current #runs in this path
  std::string runPath; // current experiment path
  std::string reloadPath; // path to model to reload
  std::string runStatus = argv[1];
  int64_t startEpoch = 0;
  int64_t startUpdate = 0;
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  if (runStatus == kTrainMode) {
    LOG(INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (!FLAGS_flagsfile.empty()) {
      LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadPath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
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
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    auto epoch = cfg.find(kEpoch);
    if (epoch == cfg.end()) {
      LOG(WARNING) << "Did not find epoch to start from, starting from 0.";
    } else {
      startEpoch = std::stoi(epoch->second);
    }
    auto nbupdates = cfg.find(kUpdates);
    if (nbupdates == cfg.end()) {
      LOG(WARNING) << "Did not find #updates to start from, starting from 0.";
    } else {
      startUpdate = std::stoi(nbupdates->second);
    }
  } else if (runStatus == kForkMode) {
    reloadPath = argv[2];
    std::unordered_map<std::string, std::string> cfg;
    W2lSerializer::load(reloadPath, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }

    LOG(INFO) << "Reading flags from config file " << reloadPath;
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
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    runPath = newRunPath(FLAGS_rundir, FLAGS_runname, FLAGS_tag);
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }
  // Only new flags are re-serialized. Copy any values from deprecated flags to
  // new flags when deprecated flags are present and corresponding new flags
  // aren't
  w2l::handleDeprecatedFlags();

  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(
        1.0 / fl::getWorldSize(), true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  LOG_MASTER(INFO) << "Experiment path: " << runPath;
  LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, getEnvVar("USER")},
      {kHostName, getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  auto validSets = split(',', trim(FLAGS_valid));
  std::vector<std::pair<std::string, std::string>> validTagSets;
  for (const auto& s : validSets) {
    // assume the format is tag:filepath
    auto ts = splitOnAnyOf(":", s);
    if (ts.size() == 1) {
      validTagSets.emplace_back(std::make_pair(s, s));
    } else {
      validTagSets.emplace_back(std::make_pair(ts[0], ts[1]));
    }
  }

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with --tokensdir and --tokens: \"" +
        dictPath + "\"");
  }
  Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry(std::to_string(r));
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  if (FLAGS_eostoken) {
    tokenDict.addEntry(kEosToken);
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  Dictionary wordDict;
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
  }

  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;

  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  if (runStatus == kTrainMode) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();
    // Encoder network, works on audio
    network = createW2lSeqModule(archfile, numFeatures, numClasses);

    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqCriterion) {
      criterion = std::make_shared<Seq2SeqCriterion>(
          buildSeq2Seq(numClasses, tokenDict.getIndex(kEosToken)));
    } else if (FLAGS_criterion == kTransformerCriterion) {
      criterion =
          std::make_shared<TransformerCriterion>(buildTransformerCriterion(
              numClasses,
              FLAGS_am_decoder_tr_layers,
              FLAGS_am_decoder_tr_dropout,
              FLAGS_am_decoder_tr_layerdrop,
              tokenDict.getIndex(kEosToken)));
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }
  } else if (runStatus == kForkMode) {
    std::unordered_map<std::string, std::string> cfg; // unused
    W2lSerializer::load(reloadPath, cfg, network, criterion);
  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused
    W2lSerializer::load(
        reloadPath, cfg, network, criterion, netoptim, critoptim);
  }
  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  }
  LOG_MASTER(INFO) << "[Network Optimizer] " << netoptim->prettyString();
  LOG_MASTER(INFO) << "[Criterion Optimizer] " << critoptim->prettyString();

  double initLinNetlr = FLAGS_linlr >= 0.0 ? FLAGS_linlr : FLAGS_lr;
  double initLinCritlr =
      FLAGS_linlrcrit >= 0.0 ? FLAGS_linlrcrit : FLAGS_lrcrit;
  std::shared_ptr<LinSegCriterion> linseg;
  std::shared_ptr<fl::FirstOrderOptimizer> linNetoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> linCritoptim;
  if (FLAGS_linseg > startUpdate) {
    if (FLAGS_criterion != kAsgCriterion) {
      LOG(FATAL) << "linseg may only be used with ASG criterion";
    }
    linseg = std::make_shared<LinSegCriterion>(numClasses, scalemode);
    linseg->setParams(criterion->param(0), 0);
    LOG_MASTER(INFO) << "[Criterion] " << linseg->prettyString()
                     << " (for first " << FLAGS_linseg - startUpdate
                     << " updates)";

    linNetoptim = initOptimizer(
        {network},
        FLAGS_netoptim,
        initLinNetlr,
        FLAGS_momentum,
        FLAGS_weightdecay);
    linCritoptim =
        initOptimizer({linseg}, FLAGS_critoptim, initLinCritlr, 0.0, 0.0);

    LOG_MASTER(INFO) << "[Network Optimizer] " << linNetoptim->prettyString()
                     << " (for first " << FLAGS_linseg - startUpdate
                     << " updates)";
    LOG_MASTER(INFO) << "[Criterion Optimizer] " << linCritoptim->prettyString()
                     << " (for first " << FLAGS_linseg - startUpdate
                     << " updates)";
  }

  /* ===================== Meters ===================== */
  TrainMeters meters;
  for (const auto& s : validTagSets) {
    meters.valid[s.first] = DatasetMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validTagSets) {
    validminerrs[s.first] = DBL_MAX;
  }

  /* ===================== Logging ===================== */
  std::ofstream logFile, perfFile;
  if (isMaster) {
    dirCreate(runPath);
    logFile.open(getRunFile("log", runIdx, runPath));
    if (!logFile.is_open()) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    perfFile.open(getRunFile("perf", runIdx, runPath));
    if (!perfFile.is_open()) {
      LOG(FATAL) << "failed to open perf file for writing";
    }
    // write perf header
    auto perfMsg = getStatus(meters, 0, 0, 0, 0, false, true, "\t").first;
    appendToLog(perfFile, "# " + perfMsg);
    // write config
    std::ofstream configFile(getRunFile("config", runIdx, runPath));
    cereal::JSONOutputArchive ar(configFile);
    ar(CEREAL_NVP(config));
  }

  auto logStatus = [&perfFile, &logFile, isMaster](
                       TrainMeters& mtrs,
                       int64_t epoch,
                       int64_t nupdates,
                       double lr,
                       double lrcrit) {
    syncMeter(mtrs);

    if (isMaster) {
      auto logMsg =
          getStatus(mtrs, epoch, nupdates, lr, lrcrit, true, false, " | ")
              .second;
      auto perfMsg =
          getStatus(mtrs, epoch, nupdates, lr, lrcrit, false, true).second;
      LOG_MASTER(INFO) << logMsg;
      appendToLog(logFile, logMsg);
      appendToLog(perfFile, perfMsg);
    }
  };

  auto saveModels = [&](int iter, int totalUpdates) {
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);
      config[kUpdates] = std::to_string(totalUpdates);

      std::string filename;
      if (FLAGS_itersave) {
        filename =
            getRunFile(format("model_iter_%03d.bin", iter), runIdx, runPath);
        W2lSerializer::save(
            filename, config, network, criterion, netoptim, critoptim);
      }

      // save last model
      filename = getRunFile("model_last.bin", runIdx, runPath);
      W2lSerializer::save(
          filename, config, network, criterion, netoptim, critoptim);

      // save if better than ever for one valid
      for (const auto& v : validminerrs) {
        double verr = meters.valid[v.first].wrdEdit.value()[0];
        if (verr < validminerrs[v.first]) {
          validminerrs[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname =
              getRunFile("model_" + cleaned_v + ".bin", runIdx, runPath);
          W2lSerializer::save(
              vfname, config, network, criterion, netoptim, critoptim);
        }
      }
      // print brief stats on memory allocation (so far)
      auto* curMemMgr =
          fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
      if (curMemMgr) {
        curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
      }
    }
  };

  /* ===================== Create Dataset ===================== */
  auto trainds = createDataset(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    trainds->shuffle(FLAGS_seed);
  }

  std::map<std::string, std::shared_ptr<W2lDataset>> validds;
  int64_t validBatchSize =
      FLAGS_validbatchsize == -1 ? FLAGS_batchsize : FLAGS_validbatchsize;
  for (const auto& s : validTagSets) {
    validds[s.first] = createDataset(
        s.second, dicts, lexicon, validBatchSize, worldRank, worldSize);
  }

  /* ===================== Hooks ===================== */
  auto evalOutput = [&dicts, &criterion](
                        const af::array& op,
                        const af::array& target,
                        DatasetMeters& mtr) {
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

      auto ltrPred = tknPrediction2Ltr(viterbipath, tgtDict);
      auto ltrTgt = tknTarget2Ltr(tgtraw, tgtDict);

      auto wrdPred = tkn2Wrd(ltrPred);
      auto wrdTgt = tkn2Wrd(ltrTgt);

      mtr.tknEdit.add(ltrPred, ltrTgt);
      mtr.wrdEdit.add(wrdPred, wrdTgt);
    }
  };

  auto test = [&evalOutput](
                  std::shared_ptr<fl::Module> ntwrk,
                  std::shared_ptr<SequenceCriterion> crit,
                  std::shared_ptr<W2lDataset> testds,
                  DatasetMeters& mtrs) {
    ntwrk->eval();
    crit->eval();
    mtrs.tknEdit.reset();
    mtrs.wrdEdit.reset();
    mtrs.loss.reset();

    for (auto& batch : *testds) {
      auto output = ntwrk->forward({fl::input(batch[kInputIdx])}).front();
      auto loss =
          crit->forward({output, fl::Variable(batch[kTargetIdx], false)})
              .front();
      mtrs.loss.add(loss.array());
      evalOutput(output.array(), batch[kTargetIdx], mtrs);
    }
  };

  auto trainEvalIds =
      getTrainEvalIds(trainds->size(), FLAGS_pcttraineval, FLAGS_seed);

  int64_t curEpoch = startEpoch;

  auto train = [&meters,
                &test,
                &logStatus,
                &saveModels,
                &evalOutput,
                &validds,
                &trainEvalIds,
                &curEpoch,
                &startUpdate,
                reducer](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<W2lDataset> trainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   std::shared_ptr<fl::FirstOrderOptimizer> critopt,
                   double initlr,
                   double initcritlr,
                   bool clampCrit,
                   int64_t nbatches) {
    if (reducer) {
      fl::distributeModuleGrads(ntwrk, reducer);
      fl::distributeModuleGrads(crit, reducer);
    }

    meters.train.loss.reset();
    meters.train.tknEdit.reset();
    meters.train.wrdEdit.reset();

    std::shared_ptr<SpecAugment> saug;
    if (FLAGS_saug_start_update >= 0) {
      saug = std::make_shared<SpecAugment>(
          FLAGS_filterbanks,
          FLAGS_saug_fmaskf,
          FLAGS_saug_fmaskn,
          FLAGS_saug_tmaskt,
          FLAGS_saug_tmaskp,
          FLAGS_saug_tmaskn);
    }

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
    auto runValAndSaveModel = [&](int64_t totalEpochs,
                                  int64_t totalUpdates,
                                  double lr,
                                  double lrcrit) {
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
      try {
        logStatus(meters, totalEpochs, totalUpdates, lr, lrcrit);
      } catch (const std::exception& ex) {
        LOG(ERROR) << "Error while writing logs: " << ex.what();
      }
      // save last and best models
      try {
        saveModels(totalEpochs, totalUpdates);
      } catch (const std::exception& ex) {
        LOG(FATAL) << "Error while saving models: " << ex.what();
      }
      // reset meters for next readings
      meters.train.loss.reset();
      meters.train.tknEdit.reset();
      meters.train.wrdEdit.reset();
    };

    int64_t curBatch = startUpdate;
    while (curBatch < nbatches) {
      ++curEpoch; // counts partial epochs too!
      int epochsAfterDecay = curEpoch - FLAGS_lr_decay;
      double lrDecayScale = std::pow(
          0.5,
          (epochsAfterDecay < 0 ? 0
                                : 1 + epochsAfterDecay / FLAGS_lr_decay_step));
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
      for (auto& batch : *trainset) {
        ++curBatch;
        double lrScheduleScale;
        if (FLAGS_lrcosine) {
          const double pi = std::acos(-1);
          lrScheduleScale =
              std::cos(((double)curBatch) / ((double)nbatches) * pi / 2.0);
        } else {
          lrScheduleScale =
              std::pow(FLAGS_gamma, (double)curBatch / (double)FLAGS_stepsize);
        }
        netopt->setLr(
            initlr * lrDecayScale * lrScheduleScale *
            std::min(curBatch / double(FLAGS_warmup), 1.0));
        critopt->setLr(
            initcritlr * lrDecayScale * lrScheduleScale *
            std::min(curBatch / double(FLAGS_warmup), 1.0));
        af::sync();
        meters.timer.incUnit();
        meters.sampletimer.stopAndIncUnit();
        meters.stats.add(batch[kInputIdx], batch[kTargetIdx]);
        if (af::anyTrue<bool>(af::isNaN(batch[kInputIdx])) ||
            af::anyTrue<bool>(af::isNaN(batch[kTargetIdx]))) {
          LOG(FATAL) << "Sample has NaN values - "
                     << join(",", readSampleIds(batch[kSampleIdx]));
        }

        // forward
        meters.fwdtimer.resume();
        auto input = fl::input(batch[kInputIdx]);
        if (FLAGS_saug_start_update >= 0 &&
            curBatch >= FLAGS_saug_start_update) {
          input = saug->forward(input);
        }
        auto output = ntwrk->forward({input}).front();
        af::sync();
        meters.critfwdtimer.resume();
        auto loss =
            crit->forward({output, fl::noGrad(batch[kTargetIdx])}).front();
        af::sync();
        meters.fwdtimer.stopAndIncUnit();
        meters.critfwdtimer.stopAndIncUnit();

        if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
          LOG(FATAL) << "Loss has NaN values. Samples - "
                     << join(",", readSampleIds(batch[kSampleIdx]));
        }
        meters.train.loss.add(loss.array());

        int64_t batchIdx = (curBatch - startUpdate - 1) % trainset->size();
        int64_t globalBatchIdx = trainset->getGlobalBatchIdx(batchIdx);
        if (trainEvalIds.find(globalBatchIdx) != trainEvalIds.end()) {
          evalOutput(output.array(), batch[kTargetIdx], meters.train);
        }

        // backward
        meters.bwdtimer.resume();
        netopt->zeroGrad();
        critopt->zeroGrad();
        loss.backward();
        if (reducer) {
          reducer->finalize();
        }
        af::sync();
        meters.bwdtimer.stopAndIncUnit();

        // optimizer
        meters.optimtimer.resume();

        // scale down gradients by batchsize
        for (const auto& p : ntwrk->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / FLAGS_batchsize;
        }
        for (const auto& p : crit->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / FLAGS_batchsize;
        }

        // clamp gradients
        if (FLAGS_maxgradnorm > 0) {
          auto params = ntwrk->params();
          if (clampCrit) {
            auto critparams = crit->params();
            params.insert(params.end(), critparams.begin(), critparams.end());
          }
          fl::clipGradNorm(params, FLAGS_maxgradnorm);
        }

        // update weights
        critopt->step();
        netopt->step();
        af::sync();
        meters.optimtimer.stopAndIncUnit();
        meters.sampletimer.resume();

        if (FLAGS_reportiters > 0 && curBatch % FLAGS_reportiters == 0) {
          runValAndSaveModel(
              curEpoch, curBatch, netopt->getLr(), critopt->getLr());
          resetTimeStatMeters();
          ntwrk->train();
          crit->train();
          meters.sampletimer.resume();
          meters.runtime.resume();
          meters.timer.resume();
        }
        if (curBatch > nbatches) {
          break;
        }
      }
      af::sync();
      if (FLAGS_reportiters == 0) {
        runValAndSaveModel(
            curEpoch, curBatch, netopt->getLr(), critopt->getLr());
      }
    }
  };

  /* ===================== Train ===================== */
  if (FLAGS_linseg - startUpdate > 0) {
    train(
        network,
        linseg,
        trainds,
        linNetoptim,
        linCritoptim,
        initLinNetlr,
        initLinCritlr,
        false /* clampCrit */,
        FLAGS_linseg - startUpdate);

    startUpdate = FLAGS_linseg;
    LOG_MASTER(INFO) << "Finished LinSeg";
  }

  auto s2s = std::dynamic_pointer_cast<Seq2SeqCriterion>(criterion);
  auto trde = std::dynamic_pointer_cast<TransformerCriterion>(criterion);
  if (FLAGS_pretrainWindow - startUpdate > 0) {
    if (!s2s && !trde) {
      LOG(FATAL) << "Window pretraining only allowed for seq2seq.";
    }
    train(
        network,
        criterion,
        trainds,
        netoptim,
        critoptim,
        FLAGS_lr,
        FLAGS_lrcrit,
        true,
        FLAGS_pretrainWindow - startUpdate);
    startUpdate = FLAGS_pretrainWindow;
    LOG_MASTER(INFO) << "Finished window pretraining.";
  }
  if (s2s) {
    s2s->clearWindow();
  } else if (trde) {
    trde->clearWindow();
  }

  train(
      network,
      criterion,
      trainds,
      netoptim,
      critoptim,
      FLAGS_lr,
      FLAGS_lrcrit,
      true /* clampCrit */,
      FLAGS_iter);

  LOG_MASTER(INFO) << "Finished training";
  return 0;
}
