/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <fmt/core.h>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/pkg/speech/runtime/runtime.h"

#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"

#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/runtime/common/Serializer.h"

#include "CPCCriterion.h"
#include "CPCSpecAugment.h"
#include "MTLLoss.h"
#include "SequentialBuilder.h"

// extra optimization hyperparameters
DECLARE_string(train2);
DEFINE_string(train2, "", "comma-separated list of supervised training data");

DECLARE_bool(twostage);
DEFINE_bool(twostage, false, "Separate stage training in CPC");

DECLARE_bool(traincontext);
DEFINE_bool(traincontext, false, "Train context network using supervised loss");

DECLARE_bool(trainencoder);
DEFINE_bool(trainencoder, false, "Train encoder using supervised loss");

DECLARE_string(pretrainmodel);
DEFINE_string(pretrainmodel, "", "Start training from pretrained model");

DECLARE_int64(supdelay);
DEFINE_int64(supdelay, 0, "Number of updates before updating supervised loss");

DECLARE_int64(unsupdates);
DEFINE_int64(unsupdates, 0, "Number of unsupervised updates in joint training");

DECLARE_int64(supdates);
DEFINE_int64(supdates, 0, "Number of supervised updates in joint training");

DECLARE_double(lr2);
DEFINE_double(lr2, 0.0, "Learning rate for supervised loss");

DECLARE_double(lrcrit2);
DEFINE_double(lrcrit2, 0.0, "Learning rate for supervised loss criterion");

DECLARE_string(netoptim2);
DEFINE_string(netoptim2, "adam", "Optimizer for supervised loss");

DECLARE_string(critoptim2);
DEFINE_string(critoptim2, "adam", "Optimizer for supervised loss criterion");

DECLARE_string(criterion2);
DEFINE_string(criterion2, "ctc", "Criterion for supervised task");

DECLARE_double(maxgradnorm2);
DEFINE_double(maxgradnorm2, 0.0, "Max grad norm for supervised loss");

DECLARE_int64(supwarmup);
DEFINE_int64(supwarmup, 0, "Warmup updates for supervised loss");

DECLARE_int64(hold);
DEFINE_int64(
    hold,
    0,
    "Number of updates to hold learning rate for unsupervised loss");

DECLARE_int64(suphold);
DEFINE_int64(
    suphold,
    0,
    "Number of updates to hold learning rate for supervised loss");

DECLARE_string(lr_sched);
DEFINE_string(lr_sched, "linear", "Learning rate scheduler");

DECLARE_double(lr_ld_final);
DEFINE_double(
    lr_ld_final,
    0.0,
    "Final LR when using linear learning rate scheduler");

DECLARE_int64(lr_step_decay);
DEFINE_int64(
    lr_step_decay,
    0,
    "Steps before decay when using exponential learning rate scheduler");

DECLARE_int64(supbatchsize);
DEFINE_int64(supbatchsize, 0, "Batch size for supervised loss");

// core wav2vec 2.0 hyperparameters
DECLARE_int64(contextdim);
DEFINE_int64(contextdim, 0, "Dimension of context hidden state in CPC");

DECLARE_int64(codedim);
DEFINE_int64(codedim, 0, "Dimension of encoder hidden state in CPC");

DECLARE_int64(mutualdim);
DEFINE_int64(mutualdim, 0, "Dimension of mutual information state in CPC");

DECLARE_double(maskprob);
DEFINE_double(maskprob, 0.15, "Probability of masked tokens in CPC");

DECLARE_int64(masklength);
DEFINE_int64(masklength, 1, "Length of masked tokens in CPC");

DECLARE_double(temperature);
DEFINE_double(temperature, 0.05, "Temperature scaling in CPC");

DECLARE_int64(nnegativesamples);
DEFINE_int64(nnegativesamples, 10, "Number of negative samples in CPC");

// spec augment
DECLARE_bool(use_saug);
DEFINE_bool(use_saug, false, "Enable saug in supervised loss");

DECLARE_int64(saug_warmup);
DEFINE_int64(saug_warmup, 0, "Grad scaling for encoder features");

DECLARE_double(saug_maskprob);
DEFINE_double(saug_maskprob, 0.05, "Probability of masked tokens");

// hyperparameter from wav2vec 2.0
// enabled by default
// default value copied from fairseq (set after extensive testing by Alexei)
DECLARE_double(dropout_feat);
DEFINE_double(dropout_feat, 0.1, "Dropout for encoder features");

DECLARE_double(grad_mult_feat);
DEFINE_double(grad_mult_feat, 0.1, "Grad scaling for encoder features");

// hyperparameter from wav2vec 2.0
// implemented, but disabled by default
// default value different from fairseq
DECLARE_double(l2_enc_pen);
DEFINE_double(l2_enc_pen, 0., "L2 enc penalty");

DECLARE_int64(freeze);
DEFINE_int64(
    freeze,
    0,
    "Number of updates to freeze context network in supervised loss update");

// mix of novel and familiar hyperparameters from wav2vec 2.0
// unimplemented, kept for compatibility reasons
DECLARE_int64(noffset);
DEFINE_int64(noffset, 0, "Prediction offset in CPC");

DECLARE_int64(nunits);
DEFINE_int64(nunits, 8, "Number of discrete units in CPC");

DECLARE_int64(npieces);
DEFINE_int64(npieces, 128, "Number of wordpieces in the discrete units in CPC");

DECLARE_int64(nbuffersamples);
DEFINE_int64(nbuffersamples, 1, "Number of buffer samples in CPC");

DECLARE_int64(maskmin);
DEFINE_int64(maskmin, 2, "Minimum number of masked tokens in CPC");

DECLARE_double(maskrandtokenprob);
DEFINE_double(
    maskrandtokenprob,
    0.,
    "Probablity of random masked tokens in CPC");

DECLARE_double(masksametokenprob);
DEFINE_double(
    masksametokenprob,
    0.,
    "Probability of same masked tokens in CPC");

DECLARE_double(mtllossweight);
DECLARE_string(mtllossmapping);

using fl::lib::fileExists;
using fl::lib::format;
using fl::lib::getCurrentDate;
using fl::lib::join;
using fl::lib::pathsConcat;
using fl::pkg::runtime::afToVector;
using fl::pkg::runtime::getRunFile;
using fl::pkg::runtime::Serializer;

using namespace fl::pkg::runtime;
using namespace fl::pkg::speech;
using namespace fl::lib;
using namespace fl::lib::text;
using namespace fl::lib::audio;
using namespace w2l;

typedef std::map<std::string, unsigned int> Mapping;

int main(int argc, char** argv) {
  fl::init();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
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
  int64_t startBatch = 0;
  int64_t supStartBatch = 0;
  int64_t unsupStartBatch = 0;
  if (argc <= 1) {
    FL_LOG(fl::FATAL) << gflags::ProgramUsage();
  }
  if (runStatus == kTrainMode) {
    FL_LOG(fl::INFO) << "Parsing command line flags";
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    if (!FLAGS_flagsfile.empty()) {
      FL_LOG(fl::INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    runPath = FLAGS_rundir;
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    FL_LOG(fl::INFO) << "reload path is " << reloadPath;
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      FL_LOG(fl::FATAL) << "Invalid config loaded from " << reloadPath;
    }
    FL_LOG(fl::INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    if (argc > 3) {
      FL_LOG(fl::INFO) << "Parsing command line flags";
      FL_LOG(fl::INFO)
          << "Overriding flags should be mutable when using `continue`";
      gflags::ParseCommandLineFlags(&argc, &argv, false);
    }
    if (!FLAGS_flagsfile.empty()) {
      FL_LOG(fl::INFO) << "Reading flags from file " << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    auto epoch = cfg.find(kEpoch);
    if (epoch == cfg.end()) {
      FL_LOG(fl::WARNING)
          << "Did not find epoch to start from, starting from 0.";
    } else {
      startEpoch = std::stoi(epoch->second);
    }
    auto nbupdates = cfg.find(kUpdates);
    if (nbupdates == cfg.end()) {
      FL_LOG(fl::WARNING)
          << "Did not find #updates to start from, starting from 0.";
    } else {
      startBatch = std::stoi(nbupdates->second);
    }
  } else if (runStatus == kForkMode) {
    reloadPath = argv[2];
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      FL_LOG(fl::FATAL) << "Invalid config loaded from " << reloadPath;
    }

    FL_LOG(fl::INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    if (argc > 3) {
      FL_LOG(fl::INFO) << "Parsing command line flags";
      FL_LOG(fl::INFO)
          << "Overriding flags should be mutable when using `fork`";
      gflags::ParseCommandLineFlags(&argc, &argv, false);
    }

    if (!FLAGS_flagsfile.empty()) {
      FL_LOG(fl::INFO) << "Reading flags from file" << FLAGS_flagsfile;
      gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
    }
    gflags::ParseCommandLineFlags(&argc, &argv, false);
    runPath = FLAGS_rundir;
  } else {
    FL_LOG(fl::FATAL) << gflags::ProgramUsage();
  }
  // Only new flags are re-serialized. Copy any values from deprecated flags to
  // new flags when deprecated flags are present and corresponding new flags
  // aren't
  handleDeprecatedFlags();
  if (!FLAGS_fl_log_level.empty()) {
    fl::Logging::setMaxLoggingLevel(fl::logLevelValue(FLAGS_fl_log_level));
  }

  fl::VerboseLogging::setMaxLoggingLevel(FLAGS_fl_vlog_level);

  af::setSeed(FLAGS_seed);
  af::setFFTPlanCacheSize(FLAGS_fftcachesize);
  fl::DynamicBenchmark::setBenchmarkMode(FLAGS_fl_benchmark_mode);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  FL_LOG_MASTER(INFO) << "Gflags after parsing \n"
                      << fl::pkg::speech::serializeGflags("; ");
  FL_LOG_MASTER(INFO) << "Experiment path: " << runPath;
  FL_LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  auto flOptimLevel = FLAGS_fl_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    FL_LOG(fl::INFO)
        << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      FL_LOG(fl::INFO) << "Mixed precision training enabled with no "
                          "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, fl::pkg::speech::serializeGflags()},
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
  auto dictPath = FLAGS_tokens;
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
  }
  Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
  }
  // ctc expects the blank label last
  if (FLAGS_criterion2 == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  bool isSeq2seqCrit = FLAGS_criterion == kSeq2SeqTransformerCriterion ||
      FLAGS_criterion == kSeq2SeqRNNCriterion;
  if (isSeq2seqCrit) {
    tokenDict.addEntry(fl::pkg::speech::kEosToken);
    tokenDict.addEntry(fl::lib::text::kPadToken);
  }
  if (FLAGS_codedim == 0 || FLAGS_contextdim == 0) {
    throw std::runtime_error("Please specify encoder and context dims");
  }

  int numClasses = tokenDict.indexSize();
  FL_LOG(fl::INFO) << "Number of classes (network): " << numClasses;
  int numQuant = FLAGS_npieces * FLAGS_nunits;
  FL_LOG(fl::INFO) << "Number of quantized tokens (network): " << numQuant;

  Dictionary wordDict;
  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = createWordDict(lexicon);
    FL_LOG(fl::INFO) << "Number of words: " << wordDict.indexSize();
  }

  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Sequential> network;
  std::shared_ptr<fl::Sequential> _network;
  std::shared_ptr<fl::Sequential> _feat_network;
  std::shared_ptr<fl::Linear> mtl_classifier;
  // unsupervised criterion
  std::shared_ptr<SequenceCriterion> criterion;
  // supervised criterion (all variables ending with 2 are supervised)
  std::shared_ptr<SequenceCriterion> criterion2;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim2;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim2;
  std::shared_ptr<fl::FirstOrderOptimizer> mtloptim;
  std::unordered_map<std::string, std::string> cfg;
  std::unordered_map<std::string, std::string> _cfg;
  std::map<std::string, unsigned int> mtl_mapping;

  FL_LOG(fl::INFO) << "SAUG";
  auto saug = std::make_shared<w2l::CPCSpecAugment>(
      FLAGS_contextdim, // default 80
      64,
      6,
      FLAGS_masklength,
      FLAGS_saug_maskprob * FLAGS_masklength,
      1);

  FL_LOG(fl::INFO) << "SAUG Done";

  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);

  FeatureParams featParams(
      FLAGS_samplerate,
      FLAGS_framesizems,
      FLAGS_framestridems,
      FLAGS_filterbanks,
      FLAGS_lowfreqfilterbank,
      FLAGS_highfreqfilterbank,
      FLAGS_mfcccoeffs,
      kLifterParam /* lifterparam */,
      FLAGS_devwin /* delta window */,
      FLAGS_devwin /* delta-delta window */);
  featParams.useEnergy = false;
  featParams.usePower = false;
  featParams.zeroMeanFrame = false;
  auto featureRes =
      getFeatureType(FLAGS_features_type, FLAGS_channels, featParams);
  int numFeatures = featureRes.first;
  FeatureType featType = featureRes.second;

  if (runStatus == kTrainMode) {
    // order of arch (network) files: encoder, context, predict
    std::vector<std::string> archfiles = split(',', trim(FLAGS_arch));
    network = std::make_shared<fl::Sequential>();

    FL_LOG(fl::INFO) << "Building the network";

    if (FLAGS_pretrainmodel.length() > 0) {
      FL_LOG(fl::INFO) << "Pretrain";
      std::string version;
      network = std::make_shared<fl::Sequential>();
      Serializer::load(FLAGS_pretrainmodel, version, _cfg, _network, criterion);
      FL_LOG(fl::INFO) << "Loaded";
      PartialLoading(-1, _network, network);
      FL_LOG(fl::INFO) << "[Criterion] " << criterion->prettyString();
    } else {
      FL_LOG(fl::INFO) << "Loading architecture file from " << archfiles[0];
      network->add(
          w2l::cpc::buildSequentialModule(
              archfiles[0], numFeatures, FLAGS_codedim));
      // 2 extra layers between encoder and context in order to perform
      // operations on
      // intermediate activations
      network->add(std::make_shared<fl::LayerNorm>(std::vector<int>{0, 3}));
      network->add(
          std::make_shared<fl::Linear>(FLAGS_codedim, FLAGS_contextdim));
      FL_LOG(fl::INFO) << "Loading architecture file from " << archfiles[1];
      network->add(
          w2l::cpc::buildSequentialModule(
              archfiles[1], FLAGS_contextdim, FLAGS_contextdim));
    }
    FL_LOG(fl::INFO) << "Loading architecture file from " << archfiles[2];
    network->add(
        w2l::cpc::buildSequentialModule(
            archfiles[2], FLAGS_contextdim, numClasses));

    if (FLAGS_criterion2 == kCtcCriterion) {
      criterion2 = std::make_shared<CTCLoss>(scalemode);
    } else {
      FL_LOG(fl::FATAL) << "unimplemented criterion";
    }
    if ((FLAGS_pretrainmodel.length() == 0) &&
        (FLAGS_criterion == kCPCCriterion)) {
      criterion = std::make_shared<CPCCriterion>(
          FLAGS_codedim,
          FLAGS_contextdim,
          FLAGS_mutualdim,
          FLAGS_noffset,
          FLAGS_nunits,
          FLAGS_npieces,
          FLAGS_nnegativesamples,
          FLAGS_nbuffersamples,
          FLAGS_temperature);
      FL_LOG(fl::INFO) << "CPC criterion loaded";
    }
  } else if (runStatus == kForkMode) {
    FL_LOG(fl::INFO) << "Fork mode";
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(reloadPath, version, cfg, network, criterion);
  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(
        reloadPath,
        version,
        cfg,
        network,
        criterion,
        criterion2,
        netoptim,
        netoptim2,
        critoptim,
        critoptim2);
  }
  FL_LOG(fl::INFO) << "[Network] " << network->prettyString();
  FL_LOG(fl::INFO) << "[Network Params: " << numTotalParams(network) << "]";
  FL_LOG(fl::INFO) << "[Criterion] " << criterion->prettyString();
  FL_LOG(fl::INFO) << "[Criterion2] " << criterion2->prettyString();

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    netoptim2 = initOptimizer(
        {network},
        FLAGS_netoptim2,
        FLAGS_lr2,
        FLAGS_momentum,
        FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
    critoptim2 =
        initOptimizer({criterion2}, FLAGS_critoptim2, FLAGS_lrcrit2, 0.0, 0.0);
  }
  FL_LOG(fl::INFO) << "[Network Optimizer] " << netoptim->prettyString();
  FL_LOG(fl::INFO) << "[Network2 Optimizer] " << netoptim2->prettyString();
  FL_LOG(fl::INFO) << "[Criterion Optimizer] " << critoptim->prettyString();
  FL_LOG(fl::INFO) << "[Criterion2 Optimizer] " << critoptim2->prettyString();

  TrainMeters meters;
  TrainMeters meters2;
  for (const auto& s : validTagSets) {
    meters.valid[s.first] = DatasetMeters();
    meters2.valid[s.first] = DatasetMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validTagSets) {
    validminerrs[s.first] = DBL_MAX;
  }
  std::unordered_map<std::string, double> validWerWithDecoder;

  /* =========== Create MTLLoss module ==================================== */
  if (!FLAGS_mtllossmapping.empty()) {
    FL_LOG(fl::INFO) << "Building the MTL Loss";
    FL_LOG(fl::INFO) << "Loading " << FLAGS_mtllossmapping;
    mtl_mapping = asr4real::loadMapping(FLAGS_mtllossmapping);
    const int n_categories = mtl_mapping.size();

    mtl_classifier =
        std::make_shared<fl::Linear>(FLAGS_contextdim, n_categories);

    // TODO : update
    mtloptim = initOptimizer(
        {mtl_classifier}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);

    FL_LOG(fl::INFO) << "[MTL Classifier] " << mtl_classifier->prettyString();
    FL_LOG(fl::INFO) << "[MTL Optimizer] " << mtloptim->prettyString();
  }

  /* ===================== Logging ===================== */
  std::ofstream logFile;
  if (isMaster) {
    dirCreate(runPath);
    logFile.open(getRunFile("log", runIdx, runPath));
    if (!logFile.is_open()) {
      FL_LOG(fl::FATAL) << "failed to open log file for writing";
    }
    // write config
    std::ofstream configFile(getRunFile("config", runIdx, runPath));
    cereal::JSONOutputArchive ar(configFile);
    ar(CEREAL_NVP(config));
  }

  auto logStatus = [&logFile, isMaster](
                       TrainMeters& mtrs,
                       std::unordered_map<std::string, double>&
                           validWerWithDecoder,
                       int64_t epoch,
                       int64_t nupdates,
                       double lr,
                       double lrcrit,
                       double scaleFactor) {
    syncMeter(mtrs);

    if (isMaster) {
      auto logMsg = getLogString(
          mtrs, validWerWithDecoder, epoch, nupdates, lr, lrcrit, scaleFactor);
      FL_LOG_MASTER(INFO) << logMsg;
      appendToLog(logFile, logMsg);
    }
  };

  auto saveModels = [&](int iter, int totalupdates, bool saveValid) {
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);
      config[kUpdates] = std::to_string(totalupdates);

      std::string filename;
      if (FLAGS_itersave) {
        filename =
            getRunFile(format("model_iter_%03d.bin", iter), runIdx, runPath);
        Serializer::save(
            filename,
            FL_APP_ASR_VERSION,
            config,
            network,
            criterion,
            criterion2,
            netoptim,
            netoptim2,
            critoptim,
            critoptim2);
      }

      // save last model
      filename = getRunFile("model_last.bin", runIdx, runPath);
      Serializer::save(
          filename,
          FL_APP_ASR_VERSION,
          config,
          network,
          criterion,
          criterion2,
          netoptim,
          netoptim2,
          critoptim,
          critoptim2);

      // save if better than ever for one valid (using supervised meters)
      for (const auto& v : validminerrs) {
        double verr;
        verr = meters2.valid[v.first].wrdEdit.errorRate()[0];

        if ((verr > 0.01) && (verr < validminerrs[v.first])) {
          validminerrs[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname =
              getRunFile("model_" + cleaned_v + ".bin", runIdx, runPath);
          Serializer::save(
              vfname,
              FL_APP_ASR_VERSION,
              config,
              network,
              criterion,
              criterion2,
              netoptim,
              netoptim2,
              critoptim,
              critoptim2);
        }
      }

      auto* curMemMgr =
          fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
      if (curMemMgr) {
        curMemMgr->printInfo("Memory Manager Stats", 0 /* device id */);
      }
    }
  };

  /* ===================== Create Dataset ===================== */

  int64_t supbatchsize = FLAGS_supbatchsize;
  if (supbatchsize == 0) {
    supbatchsize = FLAGS_batchsize;
  }

  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion2,
      FLAGS_surround,
      isSeq2seqCrit,
      FLAGS_replabel,
      true /* skip unk */,
      FLAGS_usewordpiece /* fallback2LetterWordSepLeft */,
      !FLAGS_usewordpiece /* fallback2LetterWordSepLeft */);

  const auto sfxConf = (FLAGS_sfx_config.empty())
      ? std::vector<sfx::SoundEffectConfig>()
      : sfx::readSoundEffectConfigFile(FLAGS_sfx_config);

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      sfxConf);
  auto targetTransform = targetFeatures(tokenDict, lexicon, targetGenConfig);
  auto wordTransform = wordFeatures(wordDict);
  int targetpadVal = isSeq2seqCrit
      ? tokenDict.getIndex(fl::lib::text::kPadToken)
      : kTargetPadValue;
  int wordpadVal = kTargetPadValue;
  auto padVal = std::make_tuple(0, targetpadVal, wordpadVal);

  std::vector<std::string> trainSplits = split(",", FLAGS_train, true);
  auto trainds = createDataset(
      trainSplits,
      FLAGS_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      padVal,
      worldRank,
      worldSize,
      false, // allowEmpty
      FLAGS_batching_strategy,
      FLAGS_batching_max_duration);

  std::vector<std::string> trainSplits2 = split(",", FLAGS_train2, true);
  auto trainds2 = createDataset(
      trainSplits2,
      FLAGS_datadir,
      FLAGS_batchsize,
      inputTransform,
      targetTransform,
      wordTransform,
      padVal,
      worldRank,
      worldSize,
      false, // allowEmpty
      FLAGS_batching_strategy,
      FLAGS_batching_max_duration);

  std::map<std::string, std::shared_ptr<fl::Dataset>> validds;
  int64_t validBatchSize =
      FLAGS_validbatchsize == -1 ? FLAGS_batchsize : FLAGS_validbatchsize;
  for (const auto& s : validTagSets) {
    validds[s.first] = createDataset(
        {s.second},
        FLAGS_datadir,
        validBatchSize,
        inputTransform,
        targetTransform,
        wordTransform,
        padVal,
        worldRank,
        worldSize,
        true // allowEmpty
    );
  }

  /* ===================== Hooks ===================== */
  auto evalOutput = [&dicts, &criterion2, &isSeq2seqCrit](
                        const af::array& op,
                        const af::array& target,
                        DatasetMeters& mtr) {
    auto batchsz = op.dims(2);
    for (int b = 0; b < batchsz; ++b) {
      auto tgt = target(af::span, b);
      auto viterbipath =
          afToVector<int>(criterion2->viterbiPath(op(af::span, af::span, b)));
      auto tgtraw = afToVector<int>(tgt);

      // Remove `-1`s appended to the target for batching (if any)
      auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(labellen);

      // remap actual, predicted targets for evaluating edit distance error
      if (dicts.find(kTargetIdx) == dicts.end()) {
        FL_LOG(fl::FATAL) << "Dictionary not provided for target: "
                          << kTargetIdx;
      }
      auto tgtDict = dicts.find(kTargetIdx)->second;

      auto ltrPred = tknPrediction2Ltr(
          viterbipath,
          tgtDict,
          FLAGS_criterion2,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
      auto ltrTgt = tknTarget2Ltr(
          tgtraw,
          tgtDict,
          FLAGS_criterion2,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);

      auto wrdPred = tkn2Wrd(ltrPred, FLAGS_wordseparator);
      auto wrdTgt = tkn2Wrd(ltrTgt, FLAGS_wordseparator);

      mtr.tknEdit.add(ltrPred, ltrTgt);
      mtr.wrdEdit.add(wrdPred, wrdTgt);
    }
  };
  auto cpc_criterion = std::dynamic_pointer_cast<CPCCriterion>(criterion);

  // masking function in unsuperised loss
  auto maskFunction = [&cpc_criterion](const fl::Variable& inp) {
    auto inpMasked =
        cpc_criterion->getMask(inp, FLAGS_maskprob, FLAGS_masklength);
    return inpMasked;
  };

  auto numMaskFunction = [&cpc_criterion]() {
    return cpc_criterion->numMask();
  };

  auto test = [&maskFunction, &evalOutput](
                  std::shared_ptr<fl::Sequential> ntwrk,
                  std::shared_ptr<SequenceCriterion> crit,
                  std::shared_ptr<fl::Dataset> validds,
                  DatasetMeters& mtrs,
                  bool pretrain) {
    ntwrk->eval();
    crit->eval();
    mtrs.tknEdit.reset();
    mtrs.wrdEdit.reset();
    mtrs.loss.reset();
    auto curValidSet = loadPrefetchDataset(
        validds, FLAGS_nthread, false /* shuffle */, 0 /* seed */);
    for (auto& batch : *curValidSet) {
      std::vector<fl::Variable> crit_input;
      int idx = 0;
      auto enc_out = fl::input(batch[kInputIdx]);
      enc_out = ntwrk->module(idx++)->forward({enc_out}).front();
      enc_out = ntwrk->module(idx++)->forward({enc_out}).front();
      fl::Variable enc_out_mask;
      // mask only in unsupervised loss forward pass
      if (pretrain) {
        enc_out_mask = maskFunction(enc_out);
      } else {
        enc_out_mask = enc_out;
      }
      enc_out_mask = ntwrk->module(idx++)->forward({enc_out_mask}).front();
      auto context_mask = w2l::cpc::forwardSequentialModuleWithPadMask(
          enc_out_mask, ntwrk->module(idx++), batch[kDurationIdx]);
      // target is not used in unsupervised loss
      if (pretrain) {
        crit_input = {enc_out, context_mask};
      } else {
        auto output = ntwrk->module(idx)->forward({context_mask}).front();
        crit_input = {output, fl::Variable(batch[kTargetIdx], false)};
        evalOutput(output.array(), batch[kTargetIdx], mtrs);
      }
      auto loss = crit->forward(crit_input).front();
      mtrs.loss.add(loss.array());
    }
  };

  auto lrSched = [](int64_t iter, int64_t totalIter, bool pretrain) {
    int64_t hold, warmup;
    if (!pretrain) {
      hold = FLAGS_suphold;
      warmup = FLAGS_supwarmup;
    } else {
      hold = FLAGS_hold;
      warmup = FLAGS_warmup;
    }
    double lrScale = 1;
    // lr schedulers (in normal operation: unsupervised loss uses warmup +
    // linear,
    // superised loss uses warmup + constant, ignore custom)
    if (iter <= warmup) {
      lrScale = ((double)iter) / warmup;
    } else if (FLAGS_lr_sched == "custom") {
      if (pretrain) {
        int64_t offset = 750000;
        int64_t target = 760000;
        if (iter < offset) {
          lrScale = FLAGS_lr_ld_final;
        } else if (iter < target) {
          auto lrTarget = FLAGS_lr_ld_final +
              ((1.0 - FLAGS_lr_ld_final) * (totalIter - target)) /
                  (totalIter - hold);
          lrScale = FLAGS_lr_ld_final +
              ((lrTarget - FLAGS_lr_ld_final) * (iter - offset)) /
                  (target - offset);
        } else {
          lrScale = FLAGS_lr_ld_final +
              ((1.0 - FLAGS_lr_ld_final) * (totalIter - iter)) /
                  (totalIter - hold);
        }
      } else {
      }
    } else if (FLAGS_lr_sched == "inv_sqrt") {
      hold = std::max(warmup, hold);
      if (iter > hold) {
        lrScale = std::sqrt((double)hold) / std::sqrt((double)iter);
      }
    } else if (FLAGS_lr_sched == "linear") {
      hold = std::max(warmup, hold);
      if (iter > hold) {
        lrScale = FLAGS_lr_ld_final +
            ((1.0 - FLAGS_lr_ld_final) * (totalIter - iter)) /
                (totalIter - hold);
      }
    } else if (FLAGS_lr_sched == "step") {
      hold = std::max(warmup + FLAGS_lr_step_decay, hold);
      int64_t power = 0;
      if (iter > hold) {
        power = 1 + (iter - hold) / FLAGS_lr_step_decay;
      }
      lrScale = std::pow(2, -((double)power));
    } else if (FLAGS_lr_sched == "constant") {
    } else {
      throw std::runtime_error("Invalid lr scheduler");
    }
    return lrScale;
  };

  auto trainEvalIds =
      getTrainEvalIds(trainds2->size(), FLAGS_pcttraineval, FLAGS_seed);

  if (reducer) {
    fl::distributeModuleGrads(network, reducer);
    fl::distributeModuleGrads(criterion, reducer);
    fl::distributeModuleGrads(criterion2, reducer);
  }

  fl::allReduceParameters(network);
  fl::allReduceParameters(criterion);
  fl::allReduceParameters(criterion2);

  auto resetTimeStatMeters = [](TrainMeters& mtrs) {
    mtrs.runtime.reset();
    mtrs.stats.reset();
    mtrs.sampletimer.reset();
    mtrs.fwdtimer.reset();
    mtrs.critfwdtimer.reset();
    mtrs.bwdtimer.reset();
    mtrs.optimtimer.reset();
    mtrs.timer.reset();
  };

  // shuffled datasets for supervised and unsupervised loss
  std::map<int, std::shared_ptr<fl::Dataset>> shuffleds;
  shuffleds[0] = nullptr;
  shuffleds[1] = nullptr;
  // scale counters and factors for each loss
  std::map<int, float> scaleFactors;
  std::map<int, unsigned int> scaleCounters;
  scaleFactors[0] = 0.0f;
  scaleFactors[1] = 0.0f;
  scaleCounters[0] = 1;
  scaleCounters[1] = 1;

  auto train = [&test,
                &logStatus,
                &validWerWithDecoder,
                &saveModels,
                &evalOutput,
                &maskFunction,
                &numMaskFunction,
                &saug,
                &lrSched,
                &validds,
                &trainEvalIds,
                &cpc_criterion,
                &resetTimeStatMeters,
                &startBatch,
                &isMaster,
                &shuffleds,
                &scaleFactors,
                &scaleCounters,
                &mtl_mapping,
                reducer](
                   std::shared_ptr<fl::Sequential> ntwrk,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<fl::Dataset> trainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   std::shared_ptr<fl::FirstOrderOptimizer> critopt,
                   std::shared_ptr<fl::Linear> mtlpredictor,
                   std::shared_ptr<fl::FirstOrderOptimizer> mtloptim,
                   TrainMeters& mtrs,
                   double initlr,
                   double initcritlr,
                   bool clampCrit,
                   bool pretrain,
                   int64_t& trainStartBatch,
                   int64_t nbatches) {
    auto runValAndSaveModel = [&](int64_t epoch,
                                  int64_t totalupdates,
                                  double lr,
                                  double lrcrit,
                                  bool saveValid,
                                  float scaleFactor) {
      mtrs.runtime.stop();
      mtrs.timer.stop();
      mtrs.sampletimer.stop();
      mtrs.fwdtimer.stop();
      mtrs.critfwdtimer.stop();
      mtrs.bwdtimer.stop();
      mtrs.optimtimer.stop();

      // valid
      for (auto& vds : validds) {
        test(ntwrk, crit, vds.second, mtrs.valid[vds.first], pretrain);
      }

      // print status
      try {
        logStatus(
            mtrs,
            validWerWithDecoder,
            epoch,
            totalupdates,
            lr,
            lrcrit,
            scaleFactor);
      } catch (const std::exception& ex) {
        FL_LOG(fl::FATAL) << "Error while writing logs: " << ex.what();
      }
      // save last and best models
      try {
        saveModels(epoch, totalupdates, saveValid);
      } catch (const std::exception& ex) {
        FL_LOG(fl::FATAL) << "Error while saving models: " << ex.what();
      }
      // reset meters for next readings
      mtrs.train.loss.reset();
      mtrs.train.tknEdit.reset();
      mtrs.train.wrdEdit.reset();
    };

    // trainIdx = 0 (unsupervised), 1 (supervised)
    int trainIdx = 1 - pretrain;
    // curBatch is number of updates for the current loss being computed
    // from beginning
    int64_t curBatch = trainStartBatch;
    float scaleFactor = scaleFactors[trainIdx];
    unsigned int scaleCounter = scaleCounters[trainIdx];
    if (scaleFactor == 0.0f) {
      scaleFactor =
          FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1;
    }
    unsigned int kScaleFactorUpdateInterval =
        FLAGS_fl_amp_scale_factor_update_interval;
    unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;
    double kMinScaleFactor = 2 * fl::kAmpMinimumScaleFactorValue;

    while (curBatch < nbatches) {
      ntwrk->train();
      crit->train();

      int64_t freeze = FLAGS_freeze;
      // iter is total number of updates from beginning
      int64_t iter = startBatch + (curBatch - trainStartBatch) + 1;
      int64_t totalIter = FLAGS_iter;
      if (!pretrain) {
        iter -= FLAGS_supdelay;
        totalIter -= FLAGS_supdelay;
      }
      double lrScale = lrSched(iter, totalIter, pretrain);
      netopt->setLr(lrScale * initlr);
      critopt->setLr(lrScale * initcritlr);

      auto datasize = trainset->size();
      // batchIdx is index in batch of current loss
      auto batchIdx = curBatch % datasize;
      // curEpoch is epoch of current loss
      auto curEpoch = 1 + curBatch / datasize;
      // printf("train %d %d %d\n", trainIdx, batchIdx, datasize);

      if ((shuffleds[trainIdx] == nullptr) || (batchIdx == 0)) {
        // testing different ways of shuffling with updated dataset pipeline
        shuffleds[trainIdx] = loadPrefetchDataset(
            trainset, FLAGS_nthread, true, pretrain + curEpoch);
      }

      // auto printInfo = isMaster;
      auto printInfo = curBatch < 100;

      af::sync();
      mtrs.sampletimer.resume();
      mtrs.runtime.resume();
      mtrs.timer.resume();
      const auto& batch = (shuffleds[trainIdx])->get(batchIdx);
      ++curBatch;

      af::sync();
      mtrs.timer.incUnit();
      mtrs.sampletimer.stopAndIncUnit();
      mtrs.stats.add(batch[kInputIdx], batch[kTargetIdx]);
      if (af::anyTrue<bool>(af::isNaN(batch[kInputIdx])) ||
          af::anyTrue<bool>(af::isNaN(batch[kTargetIdx]))) {
        FL_LOG(fl::FATAL) << "Sample has NaN values - "
                          << join(",", readSampleIds(batch[kSampleIdx]));
      }

      bool retrySample = false;
      do {
        retrySample = false;

        // forward
        mtrs.fwdtimer.resume();

        std::vector<fl::Variable> crit_input;
        fl::Variable output;
        fl::Variable l2_enc_out;
        auto enc_out = fl::input(batch[kInputIdx]);
        int idx = 0;
        enc_out = ntwrk->module(idx++)->forward({enc_out}).front();
        auto dtype = enc_out.type();
        l2_enc_out =
            reorder(mean((enc_out * enc_out).as(f32), {0, 1}), 2, 0, 1, 3);
        enc_out = ntwrk->module(idx++)->forward({enc_out}).front().as(dtype);
        fl::Variable enc_out_mask;
        if (pretrain) {
          enc_out_mask = maskFunction(enc_out.as(f32)).as(dtype);
          l2_enc_out = l2_enc_out * numMaskFunction();
        } else if (FLAGS_use_saug && (iter > FLAGS_saug_warmup)) {
          saug->setMaskEmbedding(cpc_criterion->getMaskEmbedding());
          enc_out_mask = saug->forward(enc_out.as(f32)).as(dtype);
        } else {
          enc_out_mask = enc_out;
        }
        enc_out_mask = ntwrk->module(idx++)->forward({enc_out_mask}).front();

        enc_out = fl::dropout(enc_out, FLAGS_dropout_feat);
        enc_out_mask = fl::dropout(enc_out_mask, FLAGS_dropout_feat);
        auto context_mask = w2l::cpc::forwardSequentialModuleWithPadMask(
            enc_out_mask, ntwrk->module(idx++), batch[kDurationIdx]);
        if (pretrain) {
          crit_input = {enc_out, context_mask};
        } else {
          output = ntwrk->module(idx)->forward({context_mask}).front().as(f32);
          crit_input = {output, fl::noGrad(batch[kTargetIdx])};
        }
        af::sync();
        mtrs.critfwdtimer.resume();
        auto loss = crit->forward(crit_input).front();

        if (mtlpredictor) {
          mtlpredictor->train();
          mtloptim->zeroGrad();
          fl::Variable mtl_loss = asr4real::mtl_step(
              context_mask,
              mtlpredictor,
              shuffleds[trainIdx],
              mtl_mapping,
              batchIdx);
          loss = loss + FLAGS_mtllossweight * mtl_loss;
        }
        // add l2 encoder output penalty term in unsupervised loss
        if (pretrain) {
          loss = loss + FLAGS_l2_enc_pen * l2_enc_out;
        }

        if (printInfo) {
          auto str = "loss " + std::to_string(curBatch);
          af::print(str.c_str(), loss.array());
        }

        af::sync();
        mtrs.fwdtimer.stopAndIncUnit();
        mtrs.critfwdtimer.stopAndIncUnit();

        if (FLAGS_fl_amp_use_mixed_precision) {
          ++scaleCounter;
          loss = loss * scaleFactor;
        }

        if (af::anyTrue<bool>(af::isNaN(loss.array())) ||
            af::anyTrue<bool>(af::isInf(loss.array()))) {
          if (FLAGS_fl_amp_use_mixed_precision &&
              scaleFactor >= kMinScaleFactor) {
            scaleFactor = scaleFactor / 2.0f;
            if (isMaster) {
              FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                         << scaleFactor;
            }
            scaleCounter = 1;
            retrySample = true;
            continue;
          } else {
            FL_LOG(fl::FATAL) << "Loss has NaN values. Samples - "
                              << join(",", readSampleIds(batch[kSampleIdx]));
          }
        }

        std::hash<std::string> hasher;
        if (!pretrain &&
            (hasher(join(",", readSampleIds(batch[kSampleIdx]))) % 100 <=
             FLAGS_pcttraineval)) {
          evalOutput(output.array(), batch[kTargetIdx], mtrs.train);
        }

        // backward
        mtrs.bwdtimer.resume();
        netopt->zeroGrad();
        critopt->zeroGrad();
        loss.backward();
        if (reducer) {
          reducer->finalize();
        }
        af::sync();
        mtrs.bwdtimer.stopAndIncUnit();

        // optimizer
        mtrs.optimtimer.resume();

        // scale down gradients by batchsize
        af::array tokenSize = af::constant(loss.dims(0), 1, f32);
        if (reducer) {
          fl::allReduce(tokenSize);
        }
        float tokenSizeScalar = tokenSize.scalar<float>();

        for (const auto& p : ntwrk->module(0)->params()) {
          // gradient of encoder is scaled and
          // only enabled in unsupervised loss or
          // if trainencoder flag is enabled in supervised loss
          if (pretrain || FLAGS_trainencoder) {
            p.grad() = p.grad() * FLAGS_grad_mult_feat;
          } else {
            p.grad() = p.grad() * 0;
          }
        }
        if (!pretrain && !FLAGS_trainencoder) {
          for (const auto& p : ntwrk->module(1)->params()) {
            p.grad() = p.grad() * 0;
          }
        }
        // gradient of context is zero if supervised loss and traincontext
        // is false
        if (!pretrain && (!FLAGS_traincontext || (iter < freeze))) {
          for (const auto& p : ntwrk->module(2)->params()) {
            p.grad() = p.grad() * 0;
          }
          for (const auto& p : ntwrk->module(3)->params()) {
            p.grad() = p.grad() * 0;
          }
        }
        int gradIdx = 0;
        for (const auto& p : ntwrk->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / (tokenSizeScalar * scaleFactor);
          if (FLAGS_fl_amp_use_mixed_precision) {
            if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
                af::anyTrue<bool>(af::isInf(p.grad().array()))) {
              if (scaleFactor >= kMinScaleFactor) {
                scaleFactor = scaleFactor / 2.0f;
                FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                           << "gradidx " << gradIdx << "\t" << "grad dims "
                           << p.grad().dims() << "\t" << scaleFactor;
                retrySample = true;
                scaleCounter = 1;
                break;
              } else {
                FL_LOG(fl::FATAL)
                    << "Gradient Loss has NaN values. Samples - "
                    << join(",", readSampleIds(batch[kSampleIdx]));
              }
            }
          }
          gradIdx++;
        }

        if (retrySample) {
          mtrs.optimtimer.stop();
          continue;
        }

        mtrs.train.loss.add((loss / scaleFactor).array());

        for (const auto& p : crit->params()) {
          if (!p.isGradAvailable()) {
            continue;
          }
          p.grad() = p.grad() / (tokenSizeScalar * scaleFactor);
        }

      } while (retrySample);

      // debugging code
      // logStatus(mtrs, curEpoch, iter, netopt->getLr(), critopt->getLr());
      // if (curBatch == 10) {
      //   resetTimeStatMeters(mtrs);
      //}

      // clamp gradients
      double maxgradnorm = FLAGS_maxgradnorm;
      if (!pretrain && FLAGS_maxgradnorm2 > 0.0) {
        maxgradnorm = FLAGS_maxgradnorm2;
      }
      if (maxgradnorm > 0) {
        auto params = ntwrk->params();
        if (clampCrit) {
          auto critparams = crit->params();
          params.insert(params.end(), critparams.begin(), critparams.end());
        }
        auto gradnorm = fl::clipGradNorm(params, maxgradnorm);
        if (printInfo) {
          std::cout << "gradnorm " << curBatch << ": " << gradnorm << std::endl;
        }
      }

      // update weights
      if (lrScale > 0) {
        critopt->step();
        netopt->step();
      }
      if (lrScale > 0 && mtlpredictor) {
        mtloptim->step();
      }
      af::sync();
      mtrs.optimtimer.stopAndIncUnit();

      if (FLAGS_fl_amp_use_mixed_precision) {
        if (printInfo) {
          std::cout << "scale factor " << curBatch << ": " << scaleFactor
                    << std::endl;
        }
        if (scaleFactor < kMaxScaleFactor) {
          if (scaleCounter % kScaleFactorUpdateInterval == 0) {
            scaleFactor *= 2;
            if (isMaster) {
              FL_VLOG(2) << "AMP: Scale factor increased. New value:\t"
                         << scaleFactor;
            }
          } else {
            // scaleFactor += 2;
          }
        }
      }

      // mtrs.sampletimer.resume();
      mtrs.runtime.stop();
      mtrs.timer.stop();

      if (FLAGS_reportiters > 0 && curBatch % FLAGS_reportiters == 0) {
        runValAndSaveModel(
            curEpoch,
            curBatch,
            netopt->getLr(),
            critopt->getLr(),
            pretrain,
            scaleFactor);
        resetTimeStatMeters(mtrs);
      }

      if ((batchIdx == (datasize - 1)) ||
          (pretrain && (iter == FLAGS_supdelay)) || (iter == totalIter)) {
        runValAndSaveModel(
            curEpoch,
            iter,
            netopt->getLr(),
            critopt->getLr(),
            pretrain,
            scaleFactor);
        resetTimeStatMeters(mtrs);
      }
      af::sync();
    }
    trainStartBatch = curBatch;
    scaleFactors[trainIdx] = scaleFactor;
    scaleCounters[trainIdx] = scaleCounter;
  };

  std::cout << " *** >>> NEW CALL TO TRAIN" << std::endl;

  // loading from a previous checkpoint
  if (startBatch < FLAGS_supdelay) {
    unsupStartBatch = startBatch;
  } else if (FLAGS_twostage) {
    unsupStartBatch = FLAGS_supdelay;
    supStartBatch = (startBatch - FLAGS_supdelay);
  } else {
    unsupStartBatch = FLAGS_supdelay +
        (startBatch - FLAGS_supdelay) * FLAGS_unsupdates /
            (FLAGS_unsupdates + FLAGS_supdates);
    supStartBatch = (startBatch - FLAGS_supdelay) * FLAGS_supdates /
        (FLAGS_unsupdates + FLAGS_supdates);
  }
  // supStartBatch is number of updates of supervised loss
  // unsupStartBatch is number of updates of unsupervised loss
  startBatch = unsupStartBatch + supStartBatch;
  fmt::print("unsup: {}, sup: {}\n", unsupStartBatch, supStartBatch);

  resetTimeStatMeters(meters);
  resetTimeStatMeters(meters2);

  // alternately iterate between unsupervised and supervised loss
  while (startBatch < FLAGS_iter) {
    // unsupervised loss updates for FLAGS_unsupdates iterations
    // if two_stage = true, then first do only unsupervised and then only
    // supervised
    // if two_stage = false, then always do unsupervised and do supervsied only
    // after FLAGS_supdelay iterations
    if (!FLAGS_twostage || (FLAGS_twostage && startBatch < FLAGS_supdelay)) {
      train(
          network,
          criterion,
          trainds,
          netoptim,
          critoptim,
          mtl_classifier,
          mtloptim,
          meters,
          FLAGS_lr,
          FLAGS_lrcrit,
          true /* clampCrit */,
          true,
          unsupStartBatch,
          unsupStartBatch + FLAGS_unsupdates);
      startBatch = unsupStartBatch + supStartBatch;
      if (FLAGS_twostage && (startBatch == FLAGS_supdelay)) {
        break;
      }
    }
    // supervised loss updates for FLAGS_supdates iterations
    if (startBatch >= FLAGS_supdelay) {
      train(
          network,
          criterion2,
          trainds2,
          netoptim2,
          critoptim2,
          mtl_classifier,
          mtloptim,
          meters2,
          FLAGS_lr2,
          FLAGS_lrcrit2,
          true /* clampCrit */,
          false,
          supStartBatch,
          supStartBatch + FLAGS_supdates);
      startBatch = unsupStartBatch + supStartBatch;
    }
  }

  FL_LOG_MASTER(INFO) << "Finished training";
  return 0;
}
