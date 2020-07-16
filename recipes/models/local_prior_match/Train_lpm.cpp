/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "recipes/models/local_prior_match/src/module/LMWrapper.h"
#include "recipes/models/local_prior_match/src/runtime/runtime.h"
#include "runtime/runtime.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec + " train [flags]\n or " + std::string() +
      " continue [directory] [flags]\n or " + std::string(argv[0]) +
      " fork [directory/model] [flags]");

  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  auto config = setFlags(argc, argv);

  int runIdx = std::stoi(config[kRunIdx]);
  std::string reloadPath = config[kReloadPath];
  std::string propPath = config[kPropPath];
  int startEpoch = std::stoi(config[kStartEpoch]);
  int startIter = std::stoi(config[kStartIter]);
  std::string runPath = config[kRunPath];
  std::string runStatus = config[kRunStatus];

  /* ================ Set up distributed environment ================ */
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

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  Dictionary amDict(dictPath);
  // Setup-specific modifications
  if (FLAGS_eostoken) {
    amDict.addEntry(kEosToken);
  }
  int numClasses = amDict.indexSize();
  LOG_MASTER(INFO) << "Number of classes (network) = " << numClasses;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, amDict});

  // Note: fairseq vocab should start with:
  // <fairseq_style> - 0 <pad> - 1, kEosToken - 2, kUnkToken - 3
  Dictionary lmDict(FLAGS_lmdict);
  lmDict.setDefaultIndex(lmDict.getIndex(kUnkToken));

  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<Seq2SeqCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;

  if (runStatus == kTrainMode) {
    auto archfile = pathsConcat(FLAGS_archdir, FLAGS_arch);
    LOG_MASTER(INFO) << "Loading architecture file from " << archfile;
    auto numFeatures = getSpeechFeatureSize();

    network = createW2lSeqModule(archfile, numFeatures, numClasses);
    criterion = std::make_shared<Seq2SeqCriterion>(
        buildSeq2Seq(numClasses, amDict.getIndex(kEosToken)));
  } else {
    std::unordered_map<std::string, std::string> cfg; // unused
    std::shared_ptr<SequenceCriterion> base_criterion;
    W2lSerializer::load(reloadPath, cfg, network, base_criterion, netoptim);
    criterion = std::dynamic_pointer_cast<Seq2SeqCriterion>(base_criterion);
  }

  // create LM
  std::shared_ptr<fl::Module> lmNetwork;
  W2lSerializer::load(FLAGS_lm, lmNetwork);
  auto dictIndexMap = genTokenDictIndexMap(amDict, lmDict);
  auto lm = std::make_shared<LMWrapper>(
      lmNetwork, dictIndexMap, lmDict.getIndex(kEosToken));

  LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();
  LOG_MASTER(INFO) << "[Criterion Params: " << numTotalParams(criterion) << "]";
  LOG_MASTER(INFO) << "[LM] " << lm->prettyString();
  LOG_MASTER(INFO) << "[LM Params: " << numTotalParams(lm) << "]";

  if (runStatus != kContinueMode) {
    netoptim = initOptimizer(
        {network, criterion},
        FLAGS_netoptim,
        FLAGS_lr,
        FLAGS_momentum,
        FLAGS_weightdecay);
  }
  LOG_MASTER(INFO) << "[Optimizer] " << netoptim->prettyString();

  /* =========== Load Proposal Network ============ */
  LOG(INFO) << "Load proposal model from " << propPath;
  std::unordered_map<std::string, std::string> propcfg; // unused
  std::shared_ptr<fl::Module> propnet;
  std::shared_ptr<SequenceCriterion> base_propcrit;
  std::shared_ptr<Seq2SeqCriterion> propcrit;

  W2lSerializer::load(propPath, propcfg, propnet, base_propcrit);
  propcrit = std::dynamic_pointer_cast<Seq2SeqCriterion>(base_propcrit);

  /* ===================== Create Dataset ===================== */
  auto pairedDs = createDataset(
      FLAGS_train, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  auto unpairedAudioDs = createDataset(
      FLAGS_trainaudio,
      dicts,
      lexicon,
      FLAGS_unpairedBatchsize,
      worldRank,
      worldSize);

  if (FLAGS_noresample) {
    LOG_MASTER(INFO) << "Shuffling trainset";
    pairedDs->shuffle(FLAGS_seed);
    unpairedAudioDs->shuffle(FLAGS_seed);
  }

  auto trainEvalIds =
      getTrainEvalIds(pairedDs->size(), FLAGS_pcttraineval, FLAGS_seed);

  auto validSets = split(',', trim(FLAGS_valid));
  std::unordered_map<std::string, std::shared_ptr<W2lDataset>> validds;
  for (const auto& s : validSets) {
    auto ts = splitOnAnyOf(":", s);
    auto setKey = ts.size() == 1 ? s : ts[0];
    auto setValue = ts.size() == 1 ? s : ts[1];

    validds[setKey] = createDataset(
        setValue, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);
  }

  /* ===================== Training Dataset Scheduler ===================== */
  DataScheduler trainDscheduler(
      {pairedDs, unpairedAudioDs},
      {kParallelData, kUnpairedAudio},
      {FLAGS_pairediter, FLAGS_audioiter},
      startEpoch + 1);

  int64_t nItersPerEpoch = FLAGS_pairediter + FLAGS_audioiter;

  /* ===================== Meters ===================== */
  SSLTrainMeters meters;
  for (const auto& s : validds) {
    meters.valid[s.first] = SSLDatasetMeters();
  }

  /* ===================== Logging ===================== */
  bool logOnEpoch = FLAGS_reportiters == 0;
  LogHelper logHelper(runIdx, runPath, isMaster, logOnEpoch);
  logHelper.saveConfig(config);
  logHelper.writeHeader(meters);

  /* ===================== Hooks ===================== */
  if (reducer) {
    fl::distributeModuleGrads(network, reducer);
    fl::distributeModuleGrads(criterion, reducer);
  }

  fl::allReduceParameters(network);
  fl::allReduceParameters(criterion);

  /* ===================== Training starts ===================== */
  int64_t curEpoch = startEpoch;
  int64_t curIter = startIter;
  bool isPairedData;
  network->train();
  criterion->train();
  lm->eval();
  propnet->eval();
  propcrit->eval();

  logHelper.saveModel("prop.bin", propcfg, propnet, propcrit);
  runEval(propnet, propcrit, validds, meters, dicts[kTargetIdx]);
  syncMeter(meters);
  double properr = avgValidErr(meters);
  LOG_MASTER(INFO) << "Initial ProposalNetwork Err = " << properr;

  resetTrainMeters(meters);

  while (curEpoch < FLAGS_iter) {
    double lrScale = std::pow(FLAGS_gamma, curEpoch / FLAGS_stepsize);
    netoptim->setLr(lrScale * FLAGS_lr);

    ++curEpoch;
    af::sync();
    meters.timer[kSampleTimer].resume();
    meters.timer[kRuntime].resume();
    meters.timer[kTimer].resume();
    LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
    LOG_MASTER(INFO) << "  Learning rate = " << netoptim->getLr();

    int scheduleIter = 0;
    while (scheduleIter < nItersPerEpoch) {
      auto sample = trainDscheduler.get();
      isPairedData = af::allTrue<bool>(sample[kDataTypeIdx] == kParallelData);
      ++curIter;
      ++scheduleIter;
      af::sync();
      int bs = isPairedData ? FLAGS_batchsize : FLAGS_unpairedBatchsize;

      meters.timer[kTimer].incUnit();
      meters.timer[kSampleTimer].stopAndIncUnit();
      meters.stats.add(sample[kInputIdx], sample[kTargetIdx]);
      if (af::anyTrue<bool>(af::isNaN(sample[kInputIdx])) ||
          af::anyTrue<bool>(af::isNaN(sample[kTargetIdx]))) {
        LOG(FATAL) << "Sample has NaN values";
      }

      // forward
      meters.timer[kFwdTimer].resume();
      auto output = network->forward({fl::input(sample[kInputIdx])}).front();
      af::sync();

      fl::Variable loss;
      fl::Variable lment;
      auto targets = fl::noGrad(sample[kTargetIdx]);
      auto tgtLen = getTargetLength(
          targets.array(), dicts[kTargetIdx].getIndex(kEosToken));
      if (isPairedData) {
        meters.timer[kCritFwdTimer].resume();
        loss = criterion->forward({output, targets}).front();

        if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
          LOG(FATAL) << "ASR loss has NaN values";
        }
        meters.train.values[kASRLoss].add(loss.array());
        meters.timer[kCritFwdTimer].stopAndIncUnit();
      } else {
        fl::Variable lmLogprob;
        meters.timer[kBeamTimer].resume();
        std::vector<std::vector<int>> paths;
        std::vector<int> hypoNums;
        auto propoutput =
            propnet->forward({fl::input(sample[kInputIdx])}).front();
        std::tie(paths, hypoNums) = batchBeamSearch(
            propoutput, propcrit, dicts[kTargetIdx].getIndex(kEosToken));
        meters.timer[kBeamTimer].stopAndIncUnit();

        auto refLen = afToVector<int>(tgtLen);
        std::tie(paths, hypoNums) = filterBeamByLength(paths, hypoNums, refLen);
        auto hypoNumsArr =
            af::array(af::dim4(hypoNums.size()), hypoNums.data());
        af::array remIdx = af::sort(af::where(hypoNumsArr));
        int remBs = remIdx.dims()[0];

        if (remBs == 0) {
          LOG(INFO) << "WARNING : using a made-up loss because of empty batch";
          tgtLen = af::constant(0, {1}, s32);
          // create a made-up loss with 0 value that is a function of
          // parameters to train, so the grad will be all 0.
          loss = criterion->forward({output, fl::noGrad(sample[kTargetIdx])})
                     .front();
          loss = 0.0 * loss;
        } else {
          targets = fl::noGrad(
              batchTarget(paths, dicts[kTargetIdx].getIndex(kEosToken)));
          tgtLen = getTargetLength(
              targets.array(), dicts[kTargetIdx].getIndex(kEosToken));

          meters.timer[kLMFwdTimer].resume();
          lmLogprob =
              fl::negate(lm->forward({targets, fl::noGrad(tgtLen)}).front());
          meters.timer[kLMFwdTimer].stopAndIncUnit();

          meters.timer[kBeamFwdTimer].resume();
          hypoNums = afToVector<int>(hypoNumsArr(remIdx));
          output =
              batchEncoderOutput(hypoNums, output(af::span, af::span, remIdx));
          loss = criterion->forward({output, targets}).front();

          auto lmRenormProb = adjustProb(lmLogprob, hypoNums, true, true);
          loss = FLAGS_lmweight * lmRenormProb * loss;
          meters.timer[kBeamFwdTimer].stopAndIncUnit();

          meters.values[kLen].add(tgtLen);
          meters.values[kNumHypos].add(static_cast<double>(paths.size()));

          lment = entropy(lmRenormProb) / static_cast<float>(hypoNums.size());
          meters.values[kLMEnt].add(lment.array());
          meters.values[kLMScore].add(lmLogprob.array());

          if (af::anyTrue<bool>(af::isNaN(loss.array()))) {
            LOG(FATAL) << "LPM loss has NaN values";
          }
          meters.values[kLPMLoss].add(loss.array());
        }
      }

      af::sync();
      meters.timer[kFwdTimer].stopAndIncUnit();
      meters.values[kFullLoss].add(loss.array());

      // compute training error rate from parallel data
      if (isPairedData) {
        auto globalBatchIdx = afToVector<int64_t>(sample[kGlobalBatchIdx]);
        if (trainEvalIds.find(globalBatchIdx[0]) != trainEvalIds.end()) {
          evalOutput(
              output.array(),
              sample[kTargetIdx],
              meters.train.edits,
              dicts[kTargetIdx],
              criterion);
        }
      }

      // backward
      meters.timer[kBwdTimer].resume();
      netoptim->zeroGrad();
      lm->zeroGrad();

      loss.backward();
      if (reducer) {
        reducer->finalize();
      }

      af::sync();
      meters.timer[kBwdTimer].stopAndIncUnit();
      meters.timer[kOptimTimer].resume();

      // scale down gradients by batchsize note that the original batchsize
      // bs is used instead of remBs, since different workers may have
      // different remBs. for the sake of simplicity we just use bs.
      for (const auto& p : network->params()) {
        if (!p.isGradAvailable()) {
          continue;
        }
        p.grad() = p.grad() / bs;
      }
      for (const auto& p : criterion->params()) {
        if (!p.isGradAvailable()) {
          continue;
        }
        p.grad() = p.grad() / bs;
      }
      if (FLAGS_maxgradnorm > 0) {
        auto params = network->params();
        auto critparams = criterion->params();
        params.insert(params.end(), critparams.begin(), critparams.end());
        fl::clipGradNorm(params, FLAGS_maxgradnorm);
      }

      netoptim->step();
      af::sync();
      meters.timer[kOptimTimer].stopAndIncUnit();
      meters.timer[kSampleTimer].resume();

      auto lengths = afToVector<int>(tgtLen);
      LOG(INFO) << "[ Epoch " << curEpoch << " ]"
                << " Iter=" << scheduleIter << " isPairedData=" << isPairedData
                << " AvgLoss=" << fl::mean(loss, {0}).scalar<float>()
                << " MinLen="
                << *std::min_element(lengths.begin(), lengths.end())
                << " MaxLen="
                << *std::max_element(lengths.begin(), lengths.end());

      // checkpoint evaluation
      if ((!logOnEpoch && curIter % FLAGS_reportiters == 0) ||
          (logOnEpoch && scheduleIter == nItersPerEpoch)) {
        stopTimeMeters(meters);
        runEval(network, criterion, validds, meters, dicts[kTargetIdx]);

        config[kEpoch] = std::to_string(curEpoch);
        config[kIteration] = std::to_string(curIter);
        std::unordered_map<std::string, double> logFields(
            {{"lr", netoptim->getLr()}});
        logHelper.logAndSaveModel(
            meters, config, network, criterion, netoptim, logFields);

        resetTrainMeters(meters);
        network->train();
        criterion->train();
        meters.timer[kSampleTimer].resume();
        meters.timer[kRuntime].resume();
        meters.timer[kTimer].resume();

        // maybe update proposal network
        double newproperr = avgValidErr(meters);
        LOG_MASTER(INFO) << "ProposalNetwork:"
                         << " new=" << newproperr << " old=" << properr;
        if ((FLAGS_propupdate == kAlways) ||
            (FLAGS_propupdate == kBetter && properr > newproperr)) {
          LOG_MASTER(INFO) << "Update proposal model to the current model";
          logHelper.saveModel("prop.bin", config, network, criterion);
          properr = newproperr;

          std::string workerPropPath = logHelper.saveModel(
              format("prop_worker%03d.bin", worldRank),
              config,
              network,
              criterion,
              nullptr, // no optimizer for the proposal model
              true);
          W2lSerializer::load(workerPropPath, propcfg, propnet, base_propcrit);
          propcrit = std::dynamic_pointer_cast<Seq2SeqCriterion>(base_propcrit);
          propnet->eval();
          propcrit->eval();
        }
      }
    }
    af::sync();
  }

  LOG_MASTER(INFO) << "Finished training";
  return 0;
}
