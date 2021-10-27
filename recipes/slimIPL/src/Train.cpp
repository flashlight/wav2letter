/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cereal/archives/json.hpp>
#include <cereal/types/unordered_map.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "flashlight/fl/contrib/contrib.h"
#include "flashlight/fl/flashlight.h"
#include "flashlight/lib/common/System.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight/lib/text/dictionary/Utils.h"
#include "flashlight/pkg/runtime/Runtime.h"
#include "flashlight/pkg/runtime/common/DistributedUtils.h"
#include "flashlight/pkg/runtime/common/SequentialBuilder.h"
#include "flashlight/pkg/runtime/common/Serializer.h"
#include "flashlight/pkg/runtime/plugin/ModulePlugin.h"
#include "flashlight/pkg/speech/augmentation/SoundEffectConfig.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/common/Flags.h"
#include "flashlight/pkg/speech/criterion/criterion.h"
#include "flashlight/pkg/speech/data/FeatureTransforms.h"
#include "flashlight/pkg/speech/data/Utils.h"
#include "flashlight/pkg/speech/decoder/DecodeMaster.h"
#include "flashlight/pkg/speech/decoder/DecodeUtils.h"
#include "flashlight/pkg/speech/decoder/TranscriptionUtils.h"
#include "flashlight/pkg/speech/runtime/runtime.h"

#include "MyLogger.h"

using fl::lib::fileExists;
using fl::lib::format;
using fl::lib::getCurrentDate;
using fl::lib::join;
using fl::lib::pathsConcat;
using fl::lib::text::CriterionType;
using fl::pkg::runtime::afToVector;
using fl::pkg::runtime::getRunFile;
using fl::pkg::runtime::Serializer;

using namespace fl::pkg::speech;

namespace {

void parseCmdLineFlagsWrapper(int argc, char** argv) {
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  // Only new flags are re-serialized. Copy any values from deprecated flags to
  // new flags when deprecated flags are present and corresponding new flags
  // aren't
  handleDeprecatedFlags();
}

// Extra flags for IPL
DEFINE_string(unsup_datadir, "", "datadir for unsupervised lists");
DEFINE_string(
    unsup_train,
    "",
    "comma-separated list of unsupervised training data");
DEFINE_int64(slimIPL_start, 0, "updates to start slimIPL");
DEFINE_string(
    slimIPL_type,
    "naive",
    "naive - relabel then bwd; cache (use from cache, label, store label); "
    "pre-cache (label, use from cache, store label); "
    "fixed-pre-cache (label, use from cache of fixed size, store label)");
DEFINE_int64(
    slimIPL_fixed_cache_updates,
    1000,
    "PL from recent number of updates (not too old), we assume that unsup >> sup");
DEFINE_string(slimIPL_cache_path, "", "path to save/load cache");
DEFINE_int64(slimIPL_sup_updates, 1, "number of supervised updates");
DEFINE_int64(slimIPL_unsup_updates, 3, "number of unsupervised updates");
DEFINE_double(
    slimIPL_fixed_cache_update_prob,
    1.,
    "probability to update fixed cache");
DEFINE_double(slimIPL_dyn_dropout, -1, "dyn dropout to set at slimIPL_start");
DEFINE_bool(slimIPL_saug, false, "use stronger saug for sup data");
DEFINE_bool(slimIPL_ema, false, "use model averaging");
DEFINE_double(slimIPL_ema_decay, 0.999, "use model averaging");
DEFINE_bool(slimIPL_use_soft, false, "use soft-labeling");
DEFINE_double(slimIPL_soft_scale, 20, "soft labeleing loss scaling");

} // namespace

int main(int argc, char** argv) {
  fl::init();
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
    parseCmdLineFlagsWrapper(argc, argv);
    runPath = FLAGS_rundir;
  } else if (runStatus == kContinueMode) {
    runPath = argv[2];
    while (fileExists(getRunFile("model_last.bin", runIdx, runPath))) {
      ++runIdx;
    }
    reloadPath = getRunFile("model_last.bin", runIdx - 1, runPath);
    LOG(INFO) << "reload path is " << reloadPath;
    std::unordered_map<std::string, std::string> cfg;
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }
    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
    parseCmdLineFlagsWrapper(argc, argv);
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
    std::string version;
    Serializer::load(reloadPath, version, cfg);
    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "Invalid config loaded from " << reloadPath;
    }

    LOG(INFO) << "Reading flags from config file " << reloadPath;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

    parseCmdLineFlagsWrapper(argc, argv);
    runPath = FLAGS_rundir;
  } else {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  if (runPath.empty()) {
    LOG(FATAL) << "'runpath' specified by --rundir, --runname cannot be empty";
  }

  af::setSeed(FLAGS_seed);
  fl::DynamicBenchmark::setBenchmarkMode(FLAGS_fl_benchmark_mode);

  std::shared_ptr<fl::Reducer> reducer = nullptr;
  if (FLAGS_enable_distributed) {
    fl::pkg::runtime::initDistributed(
        FLAGS_world_rank,
        FLAGS_world_size,
        FLAGS_max_devices_per_node,
        FLAGS_rndv_filepath);
    reducer = std::make_shared<fl::CoalescingReducer>(1.0, true, true);
  }

  int worldRank = fl::getWorldRank();
  int worldSize = fl::getWorldSize();
  bool isMaster = (worldRank == 0);

  FL_LOG_MASTER(INFO) << "Gflags after parsing \n" << serializeGflags("; ");
  FL_LOG_MASTER(INFO) << "Experiment path: " << runPath;
  FL_LOG_MASTER(INFO) << "Experiment runidx: " << runIdx;

  // flashlight optim mode
  auto flOptimLevel = FLAGS_fl_optim_mode.empty()
      ? fl::OptimLevel::DEFAULT
      : fl::OptimMode::toOptimLevel(FLAGS_fl_optim_mode);
  fl::OptimMode::get().setOptimLevel(flOptimLevel);
  if (FLAGS_fl_amp_use_mixed_precision) {
    // Only set the optim mode to O1 if it was left empty
    LOG(INFO) << "Mixed precision training enabled. Will perform loss scaling.";
    if (FLAGS_fl_optim_mode.empty()) {
      LOG(INFO) << "Mixed precision training enabled with no "
                   "optim mode specified - setting optim mode to O1.";
      fl::OptimMode::get().setOptimLevel(fl::OptimLevel::O1);
    }
  }

  std::unordered_map<std::string, std::string> config = {
      {kProgramName, exec},
      {kCommandLine, join(" ", argvs)},
      {kGflags, serializeGflags()},
      // extra goodies
      {kUserName, fl::lib::getEnvVar("USER")},
      {kHostName, fl::lib::getEnvVar("HOSTNAME")},
      {kTimestamp, getCurrentDate() + ", " + getCurrentDate()},
      {kRunIdx, std::to_string(runIdx)},
      {kRunPath, runPath}};

  std::vector<std::pair<std::string, std::string>> validTagSets =
      parseValidSets(FLAGS_valid);

  /* ===================== Create Dictionary & Lexicon ===================== */
  auto dictPath = FLAGS_tokens;
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error(
        "Invalid dictionary filepath specified with "
        " --tokens: \"" +
        dictPath + "\"");
  }
  fl::lib::text::Dictionary tokenDict(dictPath);
  // Setup-specific modifications
  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    tokenDict.addEntry("<" + std::to_string(r) + ">");
  }
  // ctc expects the blank label last
  if (FLAGS_criterion == kCtcCriterion) {
    tokenDict.addEntry(kBlankToken);
  }
  bool isSeq2seqCrit = FLAGS_criterion == kSeq2SeqTransformerCriterion ||
      FLAGS_criterion == kSeq2SeqRNNCriterion;
  if (isSeq2seqCrit) {
    tokenDict.addEntry(fl::pkg::speech::kEosToken);
    tokenDict.addEntry(fl::lib::text::kPadToken);
  }

  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  fl::lib::text::Dictionary wordDict;
  fl::lib::text::LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = fl::lib::text::loadWords(FLAGS_lexicon, FLAGS_maxword);
    wordDict = fl::lib::text::createWordDict(lexicon);
    LOG(INFO) << "Number of words: " << wordDict.indexSize();
  }

  /* ===================== Create Dataset ===================== */
  std::unordered_map<std::string, std::string> plCache;
  std::unordered_map<std::string, std::string> plCacheDump;
  std::unordered_map<std::string, af::array> plCacheSoft;
  std::unordered_map<std::string, af::array> plCacheDumpSoft;
  std::vector<int> plBatchCacheFixedSize;

  fl::lib::audio::FeatureParams featParams(
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

  TargetGenerationConfig targetGenConfig(
      FLAGS_wordseparator,
      FLAGS_sampletarget,
      FLAGS_criterion,
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

  std::vector<std::string> trainSplits = fl::lib::split(",", FLAGS_train, true);
  std::vector<std::string> unsupTrainSplits =
      fl::lib::split(",", FLAGS_unsup_train, true);
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

  std::shared_ptr<fl::Dataset> unsupTrainds;
  if (FLAGS_unsup_train != "") {
    unsupTrainds = createDataset(
        unsupTrainSplits,
        FLAGS_unsup_datadir,
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
  }

  LOG(INFO) << "Sup batches " << trainds->size();
  if (unsupTrainds != nullptr) {
    LOG(INFO) << "Unsup batches " << unsupTrainds->size();
  }
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

  /* =========== Create Network & Optimizers / Reload Snapshot ============ */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::Module> networkEMA;
  std::shared_ptr<SequenceCriterion> criterion;
  std::shared_ptr<fl::FirstOrderOptimizer> netoptim;
  std::shared_ptr<fl::FirstOrderOptimizer> critoptim;
  std::shared_ptr<fl::lib::text::LM> lm;
  std::shared_ptr<WordDecodeMaster> dm;

  auto scalemode = getCriterionScaleMode(FLAGS_onorm, FLAGS_sqnorm);
  (void)fl::pkg::runtime::ModulePlugin(FLAGS_arch);
  if (runStatus == kTrainMode) {
    FL_LOG_MASTER(INFO) << "Loading architecture file from " << FLAGS_arch;
    // Encoder network, works on audio
    network = fl::pkg::runtime::ModulePlugin(FLAGS_arch)
                  .arch(numFeatures, numClasses);
    networkEMA = network;
    if (FLAGS_slimIPL_ema) {
      networkEMA = fl::pkg::runtime::ModulePlugin(FLAGS_arch)
                       .arch(numFeatures, numClasses);
      for (size_t i = 0; i < networkEMA->params().size(); ++i) {
        auto param = network->param(i).array();
        param.eval();
        networkEMA->setParams(fl::Variable(param, false), i);
      }
    }
    if (FLAGS_criterion == kCtcCriterion) {
      criterion = std::make_shared<CTCLoss>(scalemode);
    } else if (FLAGS_criterion == kAsgCriterion) {
      criterion =
          std::make_shared<ASGLoss>(numClasses, scalemode, FLAGS_transdiag);
    } else if (FLAGS_criterion == kSeq2SeqRNNCriterion) {
      std::vector<std::shared_ptr<AttentionBase>> attentions;
      for (int i = 0; i < FLAGS_decoderattnround; i++) {
        attentions.push_back(createAttention());
      }
      criterion = std::make_shared<Seq2SeqCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::pkg::speech::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          attentions,
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_pctteacherforcing,
          FLAGS_labelsmooth,
          FLAGS_inputfeeding,
          FLAGS_samplingstrategy,
          FLAGS_gumbeltemperature,
          FLAGS_decoderrnnlayer,
          FLAGS_decoderattnround,
          FLAGS_decoderdropout);
    } else if (FLAGS_criterion == kSeq2SeqTransformerCriterion) {
      criterion = std::make_shared<TransformerCriterion>(
          numClasses,
          FLAGS_encoderdim,
          tokenDict.getIndex(fl::pkg::speech::kEosToken),
          tokenDict.getIndex(fl::lib::text::kPadToken),
          FLAGS_maxdecoderoutputlen,
          FLAGS_am_decoder_tr_layers,
          createAttention(),
          createAttentionWindow(),
          FLAGS_trainWithWindow,
          FLAGS_labelsmooth,
          FLAGS_pctteacherforcing,
          FLAGS_am_decoder_tr_dropout,
          FLAGS_am_decoder_tr_layerdrop);
    } else {
      LOG(FATAL) << "unimplemented criterion";
    }
  } else if (runStatus == kForkMode) {
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(reloadPath, version, cfg, network, criterion);
    if (version != FL_APP_ASR_VERSION) {
      LOG(WARNING) << "Model version " << version << " and code version "
                   << FL_APP_ASR_VERSION;
      // TODO fix EMA
    }
  } else { // kContinueMode
    std::unordered_map<std::string, std::string> cfg; // unused
    std::string version;
    Serializer::load(
        reloadPath, version, cfg, network, criterion, netoptim, critoptim);
    if (version != FL_APP_ASR_VERSION) {
      LOG(WARNING) << "Model version " << version << " and code version "
                   << FL_APP_ASR_VERSION;
    }
    networkEMA = network;
    if (FLAGS_slimIPL_ema) {
      Serializer::load(
          getRunFile("model_last_ema.bin", runIdx - 1, runPath),
          version,
          networkEMA);
    }
    LOG(INFO) << "Loaded model for continue training";
    if (FLAGS_slimIPL_type == "cache" || FLAGS_slimIPL_type == "pre-cache" ||
        FLAGS_slimIPL_type == "fixed-pre-cache") {
      LOG(INFO) << "Reading PL cache";
      if (FLAGS_slimIPL_use_soft) {
        std::string version;
        Serializer::load(
            getRunFile(
                "model_last_cache_soft" + std::to_string(worldRank),
                runIdx - 1,
                runPath),
            version,
            plCacheDumpSoft);
      } else {
        for (int processIdx = 0; processIdx < worldSize; processIdx++) {
          std::ifstream fcache;
          auto cacheName = getRunFile("model_last_cache", runIdx - 1, runPath) +
              std::to_string(processIdx);
          if (!fileExists(cacheName)) {
            LOG(INFO) << "Read cache from " << cacheName
                      << "; Skip, file doesn't exist";
            continue;
          }
          fcache.open(cacheName);
          std::string line;
          int count = 0;
          while (getline(fcache, line)) {
            auto tmp = fl::lib::split("|", line);
            if (tmp.size() == 0) {
              continue;
            } else if (tmp.size() == 1) {
              plCacheDump[tmp[0]] = "";
            } else {
              plCacheDump[tmp[0]] = tmp[1];
            }
            count++;
          }
          fcache.close();
          LOG(INFO) << "Read cache from " << cacheName
                    << " with number of samples " << count;
        }
        LOG(INFO) << "Reading PL cache is done; total size "
                  << plCacheDump.size();
      }
      if (FLAGS_slimIPL_type == "fixed-pre-cache") {
        std::ifstream fcacheFixed;
        auto cacheName =
            getRunFile("model_last_fixed_cache", runIdx - 1, runPath) +
            std::to_string(worldRank);
        if (!fileExists(cacheName)) {
          LOG(INFO) << "Read fixed cache from " << cacheName
                    << "; Skip, file doesn't exist";
        } else {
          fcacheFixed.open(cacheName);
          while (fcacheFixed.peek() != EOF &&
                 plBatchCacheFixedSize.size() <
                     FLAGS_slimIPL_fixed_cache_updates) {
            int val;
            fcacheFixed >> val;
            plBatchCacheFixedSize.push_back(val);
            if (val < 0) {
              LOG(FATAL) << "Wrong cache file, negative indices!!! " << val;
            }
          }
          LOG(INFO) << "Reading PL fixed cache is done; total size "
                    << plBatchCacheFixedSize.size();
          fcacheFixed.close();
        }
      }
    }
  }
  FL_LOG_MASTER(INFO) << "[Network] " << network->prettyString();
  FL_LOG_MASTER(INFO) << "[Network Params: " << numTotalParams(network) << "]";
  FL_LOG_MASTER(INFO) << "[Criterion] " << criterion->prettyString();

  if (!FLAGS_lm.empty()) {
    FL_LOG_MASTER(INFO) << "[Beam-search Decoder] Constructing language model "
                           "and beam search decoder";
    std::vector<float> dummyTransition;
    if (FLAGS_decodertype == "wrd" && FLAGS_lmtype == "kenlm" &&
        FLAGS_criterion == "ctc") {
      lm = std::make_shared<fl::lib::text::KenLM>(FLAGS_lm, wordDict);
      dm = std::make_shared<WordDecodeMaster>(
          network,
          lm,
          dummyTransition,
          true, // usePlugin
          tokenDict,
          wordDict,
          DecodeMasterTrainOptions{
              .repLabel = int32_t(FLAGS_replabel),
              .wordSepIsPartOfToken = FLAGS_usewordpiece,
              .surround = FLAGS_surround,
              .wordSep = FLAGS_wordseparator,
              .targetPadIdx = targetpadVal});
    } else {
      throw std::runtime_error(
          "Other decoders are not supported yet during training");
    }
  }

  if (runStatus == kTrainMode || runStatus == kForkMode) {
    netoptim = initOptimizer(
        {network}, FLAGS_netoptim, FLAGS_lr, FLAGS_momentum, FLAGS_weightdecay);
    critoptim =
        initOptimizer({criterion}, FLAGS_critoptim, FLAGS_lrcrit, 0.0, 0.0);
  }
  FL_LOG_MASTER(INFO) << "[Network Optimizer] " << netoptim->prettyString();
  FL_LOG_MASTER(INFO) << "[Criterion Optimizer] " << critoptim->prettyString();

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
    FL_LOG_MASTER(INFO) << "[Criterion] " << linseg->prettyString()
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

    FL_LOG_MASTER(INFO) << "[Network Optimizer] " << linNetoptim->prettyString()
                        << " (for first " << FLAGS_linseg - startUpdate
                        << " updates)";
    FL_LOG_MASTER(INFO) << "[Criterion Optimizer] "
                        << linCritoptim->prettyString() << " (for first "
                        << FLAGS_linseg - startUpdate << " updates)";
  }

  /* ===================== Meters ===================== */
  slimIPL::TrainMetersMy meters;
  for (const auto& s : validTagSets) {
    meters.valid[s.first] = DatasetMeters();
  }

  // best perf so far on valid datasets
  std::unordered_map<std::string, double> validminerrs;
  for (const auto& s : validTagSets) {
    validminerrs[s.first] = DBL_MAX;
  }

  std::unordered_map<std::string, double> validMinWerWithDecoder;
  std::unordered_map<std::string, double> validWerWithDecoder;
  if (dm) {
    for (const auto& s : validTagSets) {
      validMinWerWithDecoder[s.first] = DBL_MAX;
      validWerWithDecoder[s.first] = DBL_MAX;
    }
  }

  /* ===================== Logging ===================== */
  std::ofstream logFile;
  if (isMaster) {
    fl::lib::dirCreate(runPath);
    logFile.open(getRunFile("log", runIdx, runPath));
    if (!logFile) {
      LOG(FATAL) << "failed to open log file for writing";
    }
    // write config
    std::ofstream configFile(getRunFile("config", runIdx, runPath));
    cereal::JSONOutputArchive ar(configFile);
    ar(CEREAL_NVP(config));
  }

  /* ===================== PL Generator ===================== */
  auto tokenToWord = [&isSeq2seqCrit, &tokenDict](
                         const std::vector<int>& tokens,
                         bool isPrediction) -> std::vector<std::string> {
    std::vector<std::string> letters;
    if (isPrediction) {
      letters = tknPrediction2Ltr(
          tokens,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
    } else {
      letters = tknTarget2Ltr(
          tokens,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
    }
    return tkn2Wrd(letters, FLAGS_wordseparator);
  };

  /* ===================== Hooks ===================== */
  auto logStatus =
      [&logFile, &validTagSets, isMaster](
          slimIPL::TrainMetersMy& mtrs,
          std::unordered_map<std::string, double>& validWerWithDecoder,
          int64_t epoch,
          int64_t nupdates,
          double lr,
          double lrcrit) {
        slimIPL::syncMeter(mtrs);

        if (isMaster) {
          auto logMsg = getLogString(
              mtrs, validWerWithDecoder, epoch, nupdates, lr, lrcrit);
          FL_LOG_MASTER(INFO) << logMsg;
          appendToLog(logFile, logMsg);
        }
      };

  std::ofstream memLog;
  if (FLAGS_fl_log_mem_ops_interval > 0 && isMaster) {
    auto* curMemMgr =
        fl::MemoryManagerInstaller::currentlyInstalledMemoryManager();
    if (curMemMgr) {
      memLog.open(getRunFile("mem", runIdx, runPath));
      if (!memLog) {
        LOG(FATAL) << "failed to open memory log file="
                   << getRunFile("mem", runIdx, runPath) << " for writing";
      }
      curMemMgr->setLogStream(&memLog);
      curMemMgr->setLoggingEnabled(true);
      curMemMgr->setLogFlushInterval(FLAGS_fl_log_mem_ops_interval);
    }
  }

  auto saveModels = [&](int iter, int totalUpdates) {
    if (FLAGS_slimIPL_type == "pre-cache" || FLAGS_slimIPL_type == "cache" ||
        FLAGS_slimIPL_type == "fixed-pre-cache") {
      std::ofstream fcache, fcacheFixed;
      auto cacheName = getRunFile("model_last_cache", runIdx, runPath) +
          std::to_string(worldRank);
      if (FLAGS_slimIPL_use_soft) {
        Serializer::save(
            getRunFile(
                "model_last_cache_soft" + std::to_string(worldRank),
                runIdx,
                runPath),
            FL_APP_ASR_VERSION,
            plCacheSoft);
      } else {
        fcache.open(cacheName);
        for (auto const& element : plCache) {
          fcache << element.first << "|" << element.second << std::endl;
        }
        fcache.close();
        cacheName = getRunFile("model_last_fixed_cache", runIdx, runPath) +
            std::to_string(worldRank);
        fcacheFixed.open(cacheName);
        for (auto const& element : plBatchCacheFixedSize) {
          fcacheFixed << element << " ";
        }
        fcacheFixed.close();
      }
    }
    if (isMaster) {
      // Save last epoch
      config[kEpoch] = std::to_string(iter);
      config[kUpdates] = std::to_string(totalUpdates);

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
            netoptim,
            critoptim);
      }

      // save last model
      filename = getRunFile("model_last.bin", runIdx, runPath);
      Serializer::save(
          filename,
          FL_APP_ASR_VERSION,
          config,
          network,
          criterion,
          netoptim,
          critoptim);
      if (FLAGS_slimIPL_ema) {
        Serializer::save(
            getRunFile("model_last_ema.bin", runIdx, runPath),
            FL_APP_ASR_VERSION,
            networkEMA);
      }

      // save if better than ever for one valid
      for (const auto& v : validminerrs) {
        double verr = meters.valid[v.first].wrdEdit.errorRate()[0];
        if (verr < validminerrs[v.first]) {
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
              netoptim,
              critoptim);
        }
      }

      // save if better than ever for one valid with lm decoding
      for (const auto& v : validMinWerWithDecoder) {
        double verr = validWerWithDecoder[v.first];
        if (verr < validMinWerWithDecoder[v.first]) {
          validMinWerWithDecoder[v.first] = verr;
          std::string cleaned_v = cleanFilepath(v.first);
          std::string vfname = getRunFile(
              "model_" + cleaned_v + "_decoder.bin", runIdx, runPath);
          Serializer::save(
              vfname,
              FL_APP_ASR_VERSION,
              config,
              network,
              criterion,
              netoptim,
              critoptim);
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

  auto evalOutput = [&tokenDict, &criterion, &isSeq2seqCrit](
                        const af::array& op,
                        const af::array& target,
                        const af::array& inputSizes,
                        DatasetMeters& mtr) {
    auto batchsz = op.dims(2);
    for (int b = 0; b < batchsz; ++b) {
      auto tgt = target(af::span, b);
      auto viterbipath = afToVector<int>(
          criterion->viterbiPath(op(af::span, af::span, b), inputSizes.col(b)));
      auto tgtraw = afToVector<int>(tgt);

      // Remove `-1`s appended to the target for batching (if any)
      auto labellen = getTargetSize(tgtraw.data(), tgtraw.size());
      tgtraw.resize(labellen);

      // remap actual, predicted targets for evaluating edit distance error

      auto ltrPred = tknPrediction2Ltr(
          viterbipath,
          tokenDict,
          FLAGS_criterion,
          FLAGS_surround,
          isSeq2seqCrit,
          FLAGS_replabel,
          FLAGS_usewordpiece,
          FLAGS_wordseparator);
      auto ltrTgt = tknTarget2Ltr(
          tgtraw,
          tokenDict,
          FLAGS_criterion,
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

  auto test = [&evalOutput, &dm, &lexicon, &isSeq2seqCrit, &worldRank](
                  std::shared_ptr<fl::Module> ntwrk,
                  std::shared_ptr<SequenceCriterion> crit,
                  std::shared_ptr<fl::Dataset> validds,
                  DatasetMeters& mtrs,
                  double& dmErr) {
    ntwrk->eval();
    crit->eval();
    mtrs.tknEdit.reset();
    mtrs.wrdEdit.reset();
    mtrs.loss.reset();

    auto curValidset = loadPrefetchDataset(
        validds, FLAGS_nthread, false /* shuffle */, 0 /* seed */);

    if (dm) {
      fl::TimeMeter timer;
      timer.resume();
      LOG(INFO) << "[Beam-search decoder]   * DM: compute emissions "
                << curValidset->size();
      auto eds = dm->forward(curValidset);
      LOG(INFO) << "[Beam-search decoder]   * DM: decode";
      std::vector<double> lmweights;
      for (double lmweight = FLAGS_lmweight_low;
           lmweight <= FLAGS_lmweight_high;
           lmweight += FLAGS_lmweight_step) {
        lmweights.push_back(lmweight);
        LOG(INFO) << "LM " << lmweight;
      }
      std::vector<std::vector<int64_t>> wordEditDst(lmweights.size());
      std::vector<std::thread> threads;
      for (int i = 0; i < lmweights.size(); i++) {
        threads.push_back(std::thread(
            [&lmweights, &wordEditDst, &dm, &eds, &lexicon, i, worldRank]() {
              af::setDevice(worldRank % 8);
              double lmweight = lmweights[i];
              DecodeMasterLexiconOptions opt = {
                  .beamSize = FLAGS_beamsize,
                  .beamSizeToken = FLAGS_beamsizetoken,
                  .beamThreshold = FLAGS_beamthreshold,
                  .lmWeight = lmweight,
                  .silScore = FLAGS_silscore,
                  .wordScore = FLAGS_wordscore,
                  .unkScore = FLAGS_unkscore,
                  .logAdd = FLAGS_logadd,
                  .silToken = FLAGS_wordseparator,
                  .blankToken = kBlankToken,
                  .unkToken = fl::lib::text::kUnkToken,
                  .smearMode =
                      (FLAGS_smearing == "max"
                           ? fl::lib::text::SmearingMode::MAX
                           : fl::lib::text::SmearingMode::NONE)};
              auto pds = dm->decode(eds, lexicon, opt);
              // return token distance and word distance stats
              wordEditDst[i] = dm->computeMetrics(pds).second;
            }));
      }
      for (auto& thread : threads) {
        thread.join();
      }
      dmErr = DBL_MAX;
      for (int i = 0; i < lmweights.size(); i++) {
        af::array currentEditDist =
            af::constant((long long)(wordEditDst[i][0]), af::dim4(1, 1, 1, 1));
        af::array currentTokens =
            af::constant((long long)(wordEditDst[i][1]), af::dim4(1, 1, 1, 1));
        if (FLAGS_enable_distributed) {
          fl::allReduce(currentEditDist);
          fl::allReduce(currentTokens);
        }
        double wer = (double)currentEditDist.scalar<long long>() /
            currentTokens.scalar<long long>() * 100.0;
        FL_LOG_MASTER(INFO)
            << "[Beam-search decoder]   * DM: lmweight=" << lmweights[i]
            << " WER: " << wer;
        dmErr = std::min(dmErr, wer);
      }
      FL_LOG_MASTER(INFO) << "[Beam-search decoder]   * DM: done with best WER "
                          << dmErr;
      timer.stop();
      FL_LOG_MASTER(INFO)
          << "[Beam-search decoder] time spent on grid-search for decoding: "
          << timer.value() << "s";
    }

    for (auto& batch : *curValidset) {
      fl::Variable output = ntwrk
                                ->forward(
                                    {fl::input(batch[kInputIdx]),
                                     fl::noGrad(batch[kDurationIdx])})
                                .front();
      std::vector<fl::Variable> critArgs = {
          output, fl::Variable(batch[kTargetIdx], false)};
      if (isSeq2seqCrit) {
        critArgs.push_back(fl::Variable(batch[kDurationIdx], false));
        critArgs.push_back(fl::Variable(batch[kTargetSizeIdx], false));
      }
      auto loss = crit->forward(critArgs).front();
      mtrs.loss.add(loss.array());
      evalOutput(output.array(), batch[kTargetIdx], batch[kDurationIdx], mtrs);
    }
  };

  int64_t curEpoch = startEpoch;
  auto train = [&meters,
                &validWerWithDecoder,
                &test,
                &logStatus,
                &saveModels,
                &evalOutput,
                &validds,
                &curEpoch,
                &startUpdate,
                &isSeq2seqCrit,
                &targetTransform,
                &tokenToWord,
                &targetpadVal,
                &plCache,
                &plCacheDump,
                &plCacheSoft,
                &plCacheDumpSoft,
                &plBatchCacheFixedSize,
                &lexicon,
                &tokenDict,
                &wordDict,
                &worldRank,
                reducer](
                   std::shared_ptr<fl::Module> ntwrk,
                   std::shared_ptr<fl::Module> ntwrkEMA,
                   std::shared_ptr<SequenceCriterion> crit,
                   std::shared_ptr<fl::Dataset> trainset,
                   std::shared_ptr<fl::Dataset> unsupTrainset,
                   std::shared_ptr<fl::FirstOrderOptimizer> netopt,
                   std::shared_ptr<fl::FirstOrderOptimizer> critopt,
                   double initlr,
                   double initcritlr,
                   bool clampCrit,
                   int64_t nbatches) {
    fl::EditDistanceMeter unsupQuality;
    meters.train.loss.reset();
    meters.trainUnsup.loss.reset();
    meters.train.tknEdit.reset();
    meters.train.wrdEdit.reset();
    meters.trainUnsup.tknEdit.reset();
    meters.trainUnsup.wrdEdit.reset();

    std::shared_ptr<fl::Module> saug;
    std::shared_ptr<fl::Module> saugUnsup;
    if (FLAGS_saug_start_update >= 0) {
      if (FLAGS_features_type == kFeaturesRaw) {
        saugUnsup = std::make_shared<fl::RawWavSpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn,
            FLAGS_filterbanks,
            FLAGS_lowfreqfilterbank,
            FLAGS_highfreqfilterbank,
            FLAGS_samplerate);
      } else {
        saugUnsup = std::make_shared<fl::SpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn);
      }
    }

    if (FLAGS_slimIPL_saug) {
      if (FLAGS_features_type == kFeaturesRaw) {
        saug = std::make_shared<fl::RawWavSpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn,
            FLAGS_filterbanks,
            FLAGS_lowfreqfilterbank,
            FLAGS_highfreqfilterbank,
            FLAGS_samplerate);
      } else {
        saug = std::make_shared<fl::SpecAugment>(
            FLAGS_filterbanks,
            FLAGS_saug_fmaskf,
            FLAGS_saug_fmaskn + 1,
            FLAGS_saug_tmaskt,
            FLAGS_saug_tmaskp,
            FLAGS_saug_tmaskn * 1.5);
      }
    } else {
      saug = saugUnsup;
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
        double decodedWer;
        test(ntwrk, crit, vds.second, meters.valid[vds.first], decodedWer);
        if (validWerWithDecoder.find(vds.first) != validWerWithDecoder.end()) {
          validWerWithDecoder[vds.first] = decodedWer;
        }
      }

      // print status
      try {
        logStatus(
            meters, validWerWithDecoder, totalEpochs, totalUpdates, lr, lrcrit);
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
      meters.trainUnsup.loss.reset();
      meters.trainUnsup.tknEdit.reset();
      meters.trainUnsup.wrdEdit.reset();
    };

    int64_t curBatch = startUpdate;
    double scaleFactor =
        FLAGS_fl_amp_use_mixed_precision ? FLAGS_fl_amp_scale_factor : 1.;
    unsigned int kScaleFactorUpdateInterval =
        FLAGS_fl_amp_scale_factor_update_interval;
    unsigned int kMaxScaleFactor = FLAGS_fl_amp_max_scale_factor;
    unsigned short scaleCounter = 1;
    bool useUnsup = !(unsupTrainset == nullptr);
    FL_LOG_MASTER(INFO) << "Unsup is in use " << useUnsup;
    int fixedCacheIndexToLabel = -1;
    std::vector<int> unsupBatchesIndices;
    if (useUnsup) {
      unsupBatchesIndices = std::vector<int>(unsupTrainset->size(), 0);
      std::iota(unsupBatchesIndices.begin(), unsupBatchesIndices.end(), 0);
    }
    int cacheHits =
        plBatchCacheFixedSize.size() < FLAGS_slimIPL_fixed_cache_updates
        ? plBatchCacheFixedSize.size()
        : FLAGS_slimIPL_fixed_cache_updates;
    std::shared_ptr<fl::Dataset> curUnsupTrainset, curUnsupTrainsetNext;
    if (FLAGS_slimIPL_type == "fixed-pre-cache" &&
        plBatchCacheFixedSize.size() >= FLAGS_slimIPL_fixed_cache_updates) {
      std::random_shuffle(
          plBatchCacheFixedSize.begin(), plBatchCacheFixedSize.end());
      auto permfn = [plBatchCacheFixedSize](int64_t x) {
        return plBatchCacheFixedSize.at(x);
      };
      curUnsupTrainset = std::make_shared<fl::ResampleDataset>(
          unsupTrainset, permfn, plBatchCacheFixedSize.size());
      curUnsupTrainset = loadPrefetchDataset(
          curUnsupTrainset,
          FLAGS_nthread,
          false /* shuffle */,
          curBatch /* seed */);
    }
    while (curBatch < nbatches) {
      ++curEpoch; // counts partial epochs too!
      int64_t epochsAfterDecay = curEpoch - FLAGS_lr_decay;
      double lrDecayScale = std::pow(
          0.5,
          (epochsAfterDecay < 0 ? 0
                                : 1 + epochsAfterDecay / FLAGS_lr_decay_step));
      ntwrk->train();
      crit->train();
      if (FLAGS_reportiters == 0) {
        resetTimeStatMeters();
      }
      std::hash<std::string> hasher;
      FL_LOG_MASTER(INFO) << "Shuffling trainset";
      auto curTrainset = loadPrefetchDataset(
          trainset, FLAGS_nthread, true /* shuffle */, curEpoch /* seed */);
      if (useUnsup) {
        if (FLAGS_slimIPL_type != "fixed-pre-cache") {
          FL_LOG_MASTER(INFO) << "Shuffling unsup trainset";
          curUnsupTrainset = loadPrefetchDataset(
              unsupTrainset,
              FLAGS_nthread,
              true /* shuffle */,
              curBatch /* seed */);
        } else {
          FL_LOG_MASTER(INFO) << "Preparing next unsup trainset";
          std::random_shuffle(
              unsupBatchesIndices.begin(), unsupBatchesIndices.end());
          auto permfn = [unsupBatchesIndices](int64_t x) {
            return unsupBatchesIndices[x % unsupBatchesIndices.size()];
          };
          curUnsupTrainsetNext =
              std::make_shared<fl::ResampleDataset>(unsupTrainset, permfn);
          curUnsupTrainsetNext = loadPrefetchDataset(
              curUnsupTrainsetNext,
              FLAGS_nthread,
              false /* shuffle */,
              curBatch /* seed */);
        }
      }
      af::sync();
      meters.sampletimer.resume();
      meters.runtime.resume();
      meters.timer.resume();
      FL_LOG_MASTER(INFO) << "Epoch " << curEpoch << " started!";
      int unsupBatchIdx = 0, supBatchIdx = 0, setsOrderIdx = 0;
      std::vector<bool> setsOrder;
      int unsupSteps = useUnsup ? FLAGS_slimIPL_unsup_updates : 0;
      for (int index = 0; index < FLAGS_slimIPL_sup_updates + unsupSteps;
           index++) {
        if (index < FLAGS_slimIPL_sup_updates) {
          setsOrder.push_back(true);
        } else {
          setsOrder.push_back(false);
        }
      }
      std::random_shuffle(setsOrder.begin(), setsOrder.end());
      while (supBatchIdx < curTrainset->size()) {
        ++curBatch;
        std::vector<af::array> batch;
        bool isSupBatch = setsOrder[setsOrderIdx];
        std::vector<std::string> plTextArrayPreCacheToSave;
        bool fixedCacheRelabel = true;
        if (isSupBatch) {
          batch = curTrainset->get(supBatchIdx % curTrainset->size());
          LOG(INFO) << "Sup batch " << curBatch << " | " << supBatchIdx << " | "
                    << batch[kInputIdx].dims();
          ++supBatchIdx;
        } else {
          if (FLAGS_slimIPL_type == "fixed-pre-cache") {
            float rNumber =
                static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX);
            if (plBatchCacheFixedSize.size() <
                    FLAGS_slimIPL_fixed_cache_updates ||
                rNumber < FLAGS_slimIPL_fixed_cache_update_prob) {
              fixedCacheIndexToLabel++;
              fixedCacheRelabel = true;
            } else {
              fixedCacheRelabel = false;
            }
            if (fixedCacheIndexToLabel >= unsupBatchesIndices.size()) {
              fixedCacheIndexToLabel = 0;
              std::random_shuffle(
                  unsupBatchesIndices.begin(), unsupBatchesIndices.end());
              auto permfn = [unsupBatchesIndices](int64_t x) {
                return unsupBatchesIndices[x % unsupBatchesIndices.size()];
              };
              curUnsupTrainsetNext =
                  std::make_shared<fl::ResampleDataset>(unsupTrainset, permfn);
              curUnsupTrainsetNext = loadPrefetchDataset(
                  curUnsupTrainsetNext,
                  FLAGS_nthread,
                  false /* shuffle */,
                  curBatch /* seed */);
            }
            if (cacheHits == FLAGS_slimIPL_fixed_cache_updates) {
              // we read the whole cache, time to take another one
              cacheHits = 0;
              std::random_shuffle(
                  plBatchCacheFixedSize.begin(), plBatchCacheFixedSize.end());
              auto permfn = [plBatchCacheFixedSize](int64_t x) {
                return plBatchCacheFixedSize.at(x);
              };
              curUnsupTrainset = std::make_shared<fl::ResampleDataset>(
                  unsupTrainset, permfn, plBatchCacheFixedSize.size());
              curUnsupTrainset = loadPrefetchDataset(
                  curUnsupTrainset,
                  FLAGS_nthread,
                  false /* shuffle */,
                  curBatch /* seed */);
            }
            if (plBatchCacheFixedSize.size() >=
                FLAGS_slimIPL_fixed_cache_updates) {
              batch =
                  curUnsupTrainset->get(cacheHits % curUnsupTrainset->size());

              if (fixedCacheRelabel) {
                if (unsupBatchesIndices[fixedCacheIndexToLabel] < 0) {
                  LOG(FATAL)
                      << "Error in the index which wll be saved into cache";
                }
                plBatchCacheFixedSize[cacheHits] =
                    unsupBatchesIndices[fixedCacheIndexToLabel];
              }
              LOG(INFO) << "Unsup batch " << curBatch << " | " << cacheHits
                        << " | " << batch[kInputIdx].dims() << " update cache "
                        << fixedCacheRelabel;
            } else {
              if (unsupBatchesIndices[fixedCacheIndexToLabel] < 0) {
                LOG(FATAL)
                    << "Error in the index while preparing first state of cache";
              }
              plBatchCacheFixedSize.push_back(
                  unsupBatchesIndices[fixedCacheIndexToLabel]);
              LOG(INFO)
                  << "Skip usage of unsup batch as fixed cache is not ready "
                  << curBatch;
            }
            cacheHits++;
          } else {
            if (unsupBatchIdx < 0) {
              LOG(FATAL) << "index data is negative "
                         << "unsupBatchIdx" << unsupBatchIdx;
            }
            batch =
                curUnsupTrainset->get(unsupBatchIdx % curUnsupTrainset->size());
            LOG(INFO) << "Unsup batch " << curBatch << " | " << unsupBatchIdx
                      << " | " << batch[kInputIdx].dims();
            ++unsupBatchIdx;
            if (unsupBatchIdx >= curUnsupTrainset->size()) {
              unsupBatchIdx = 0;
              FL_LOG_MASTER(INFO) << "Shuffling unsup trainset";
              curUnsupTrainset = loadPrefetchDataset(
                  unsupTrainset,
                  FLAGS_nthread,
                  true /* shuffle */,
                  curBatch /* seed */);
            }
          }
        }
        setsOrderIdx++;
        if (setsOrderIdx >= setsOrder.size()) {
          setsOrderIdx = 0;
          std::random_shuffle(setsOrder.begin(), setsOrder.end());
        }
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
        if (isSupBatch) {
          meters.stats.add(batch[kDurationIdx], batch[kTargetSizeIdx]);
        }
        if (!batch.empty() &&
            (af::anyTrue<bool>(af::isNaN(batch[kInputIdx])) ||
             af::anyTrue<bool>(af::isNaN(batch[kTargetIdx])))) {
          LOG(FATAL) << "Sample has NaN values - "
                     << join(",", readSampleIds(batch[kSampleIdx]));
        }

        auto predictPLCommon = [&](std::vector<af::array> inpBatch)
            -> std::pair<std::vector<std::string>, af::array> {
          ntwrkEMA->eval();
          crit->eval();
          auto outputUnsupOriginal =
              ntwrkEMA
                  ->forward(
                      {fl::input(inpBatch[kInputIdx]),
                       fl::noGrad(inpBatch[kDurationIdx])})
                  .front()
                  .array();
          std::vector<std::vector<int>> tokenPredictions;
          auto viterbiPath =
              crit->viterbiPath(outputUnsupOriginal, inpBatch[kDurationIdx]);
          for (int index = 0; index < viterbiPath.dims(1); index++) {
            tokenPredictions.push_back(afToVector<int>(viterbiPath.col(index)));
          }
          ntwrkEMA->train();
          crit->train();
          unsupQuality.reset();

          if (curBatch % 100 == 0) {
            FL_LOG_MASTER(INFO)
                << "PL for samples "
                << join(",", readSampleIds(inpBatch[kSampleIdx]));
          }
          std::vector<std::string> plTextArray;
          for (int index = 0; index < tokenPredictions.size(); index++) {
            auto tokenPrediction = tokenPredictions[index];
            auto plArray = tokenToWord(tokenPrediction, true);
            auto plTrueArray = tokenToWord(
                afToVector<int>(inpBatch[kTargetIdx].col(index)), false);
            auto plText = fl::lib::join(" ", plArray);
            if (curBatch % 100 == 0) {
              FL_LOG_MASTER(INFO) << "PL for index " << index << ": " << plText;
            }
            unsupQuality.add(plArray, plTrueArray);
            plTextArray.push_back(plText);
          }
          fl::pkg::runtime::syncMeter(unsupQuality);
          if (fl::getWorldRank() == 0) {
            std::cout << "PL Quality for Batch " << curBatch << " : "
                      << unsupQuality.errorRate()[0] << std::endl;
          }
          return {plTextArray, outputUnsupOriginal};
        };

        auto predictPL =
            [&](std::vector<af::array> inpBatch) -> std::vector<std::string> {
          return predictPLCommon(inpBatch).first;
        };
        auto predictSoftPL = [&](std::vector<af::array> inpBatch) -> af::array {
          return predictPLCommon(inpBatch).second;
        };

        // Ensure no samples are skipped while adjusting the loss scale factor.
        // When gradient values are Inf/NaN, the model update is skipped and the
        // scale factor is adjusted accordingly for determinism.
        // The AMP algorithm implemented here mirrors:
        // - https://arxiv.org/abs/1710.03740
        // - https://bit.ly/35F5GqX
        // - https://bit.ly/3mn2qr0
        bool retrySample = false;
        bool doUpdate = true;
        std::vector<std::string> samplesIndices;
        if (!batch.empty()) {
          samplesIndices = readSampleIds(batch[kSampleIdx]);
        }
        do {
          retrySample = false;
          std::vector<fl::Variable> critArgs;
          fl::Variable input, output;
          af::array newUnsupDuration;
          // forward
          meters.fwdtimer.resume();
          if (!batch.empty()) {
            if (batch[kInputIdx].dims(3) > 60) {
              int length = 59;
              LOG(INFO) << "Shrink batch, too huge " << batch[kInputIdx].dims()
                        << " | " << batch[kDurationIdx].dims() << " | "
                        << batch[kTargetIdx].dims() << " | "
                        << batch[kSampleIdx].dims();
              batch[kInputIdx] = af::reorder(
                  af::reorder(batch[kInputIdx], 0, 3, 2, 1).cols(0, length),
                  0,
                  3,
                  2,
                  1);
              batch[kDurationIdx] = batch[kDurationIdx].cols(0, length);
              batch[kTargetIdx] = batch[kTargetIdx].cols(0, length);
              batch[kSampleIdx] = batch[kSampleIdx].cols(0, length);
            }
            input = fl::input(batch[kInputIdx]);
            if (FLAGS_saug_start_update >= 0 &&
                curBatch >= FLAGS_saug_start_update) {
              if (isSupBatch) {
                input = saug->forward({input}).front();
              } else {
                input = saugUnsup->forward({input}).front();
              }
            }
            std::vector<fl::Variable> fwdParams = {
                input, fl::noGrad(batch[kDurationIdx])};
            if (FLAGS_slimIPL_dyn_dropout >= 0 && useUnsup) {
              fwdParams.push_back(fl::noGrad(
                  af::constant(FLAGS_slimIPL_dyn_dropout, af::dim4(1))));
            }
            output = ntwrk->forward(fwdParams).front();
          }
          if (isSupBatch) {
            critArgs = {output, fl::Variable(batch[kTargetIdx], false)};
            if (isSeq2seqCrit) {
              critArgs.push_back(fl::Variable(batch[kDurationIdx], false));
              critArgs.push_back(fl::Variable(batch[kTargetSizeIdx], false));
            }
          } else {
            // unsup Batch
            std::vector<std::string> plTextArray;
            std::vector<af::array> plSoftArray;
            std::vector<int> useUnsupSamplesIndices;
            if (FLAGS_slimIPL_use_soft &&
                FLAGS_slimIPL_type == "fixed-pre-cache") {
              if (!batch.empty()) {
                for (int index = 0; index < samplesIndices.size(); index++) {
                  if (plCacheSoft.find(samplesIndices[index]) ==
                          plCacheSoft.end() &&
                      plCacheDumpSoft.find(samplesIndices[index]) !=
                          plCacheDumpSoft.end()) {
                    LOG(INFO)
                        << "Reuse extra loaded soft cache for sample "
                        << samplesIndices[index] << " for batch " << curBatch;
                    plCacheSoft[samplesIndices[index]] =
                        plCacheDumpSoft[samplesIndices[index]];
                  }
                }
                for (int index = 0; index < samplesIndices.size(); index++) {
                  if (plCacheDumpSoft.find(samplesIndices[index]) !=
                      plCacheDumpSoft.end()) { // place to add filtering too
                    useUnsupSamplesIndices.push_back(index);
                    plSoftArray.push_back(
                        plCacheDumpSoft[samplesIndices[index]]);
                  }
                }
              }
              if (FLAGS_slimIPL_type == "fixed-pre-cache" &&
                  fixedCacheRelabel) {
                if (unsupBatchIdx < 0) {
                  LOG(FATAL)
                      << "index data is negative "
                      << "fixedCacheIndexToLabel" << fixedCacheIndexToLabel;
                }
                auto nextBatch = curUnsupTrainsetNext->get(
                    fixedCacheIndexToLabel % curUnsupTrainsetNext->size());
                auto samplesIndicesNext = readSampleIds(nextBatch[kSampleIdx]);
                auto plSoftArrayPreCache = predictSoftPL(nextBatch);
                for (int index = 0; index < plSoftArrayPreCache.dims(2);
                     index++) {
                  plCacheSoft[samplesIndicesNext[index]] =
                      plSoftArrayPreCache(af::span, af::span, index);
                }
              }

              if (useUnsupSamplesIndices.size() > 0) {
                af::array maskedSamples = af::array(
                    af::dim4(useUnsupSamplesIndices.size()),
                    useUnsupSamplesIndices.data());
                std::vector<fl::Variable> newTargets;
                for (int index = 0; index < useUnsupSamplesIndices.size();
                     index++) {
                  newTargets.push_back(fl::Variable(
                      plCacheSoft
                          [samplesIndices[useUnsupSamplesIndices[index]]],
                      false));
                }
                meters.stats.add(
                    batch[kDurationIdx](maskedSamples),
                    batch[kTargetSizeIdx](maskedSamples));
                critArgs = {
                    output(af::span, af::span, maskedSamples, af::span, true),
                    fl::concatenate(newTargets, 2)};
                if (isSeq2seqCrit) {
                  critArgs.push_back(
                      fl::Variable(batch[kDurationIdx](maskedSamples), false));
                  critArgs.push_back(fl::Variable(
                      batch[kTargetSizeIdx](maskedSamples), false));
                }
              } else {
                LOG(INFO)
                    << "Skip unsupervised part of data as PL are not available yet";
              }
            } else {
              if (FLAGS_slimIPL_type == "naive") {
                // generate by current NN the labels;
                plTextArray = predictPL(batch);
                useUnsupSamplesIndices = std::vector<int>(plTextArray.size());
                std::iota(
                    useUnsupSamplesIndices.begin(),
                    useUnsupSamplesIndices.end(),
                    0);
              } else if (
                  FLAGS_slimIPL_type == "cache" ||
                  FLAGS_slimIPL_type == "pre-cache" ||
                  (FLAGS_slimIPL_type == "fixed-pre-cache" && !batch.empty())) {
                for (int index = 0; index < samplesIndices.size(); index++) {
                  if (plCache.find(samplesIndices[index]) == plCache.end() &&
                      plCacheDump.find(samplesIndices[index]) !=
                          plCacheDump.end()) {
                    LOG(INFO)
                        << "Reuse extra loaded cache for sample "
                        << samplesIndices[index] << " for batch " << curBatch;
                    plCache[samplesIndices[index]] =
                        plCacheDump[samplesIndices[index]];
                  }
                }

                for (int index = 0; index < samplesIndices.size(); index++) {
                  if (plCache.find(samplesIndices[index]) !=
                      plCache.end()) { // place to add filtering too
                    useUnsupSamplesIndices.push_back(index);
                    plTextArray.push_back(plCache[samplesIndices[index]]);
                  }
                }
                if (FLAGS_slimIPL_type == "pre-cache" ||
                    useUnsupSamplesIndices.size() == 0) {
                  // update Cache before doing model update
                  plTextArrayPreCacheToSave = predictPL(batch);
                }
              }
              if (FLAGS_slimIPL_type == "fixed-pre-cache" &&
                  fixedCacheRelabel) {
                if (unsupBatchIdx < 0) {
                  LOG(FATAL)
                      << "index data is negative "
                      << "fixedCacheIndexToLabel" << fixedCacheIndexToLabel;
                }
                auto nextBatch = curUnsupTrainsetNext->get(
                    fixedCacheIndexToLabel % curUnsupTrainsetNext->size());
                auto samplesIndicesNext = readSampleIds(nextBatch[kSampleIdx]);
                auto plTextArrayPreCache = predictPL(nextBatch);
                for (int index = 0; index < plTextArrayPreCache.size();
                     index++) {
                  plCache[samplesIndicesNext[index]] =
                      plTextArrayPreCache[index];
                }
              }

              if (useUnsupSamplesIndices.size() > 0) {
                af::array maskedSamples = af::array(
                    af::dim4(useUnsupSamplesIndices.size()),
                    useUnsupSamplesIndices.data());
                std::vector<af::array> newTargets, newTargetsSize;
                for (auto& plText : plTextArray) {
                  std::vector<char> curTarget(plText.begin(), plText.end());
                  auto target = targetTransform(
                      static_cast<void*>(curTarget.data()),
                      {static_cast<dim_t>(curTarget.size())},
                      af::dtype::b8);

                  newTargets.push_back(target);
                  newTargetsSize.push_back(
                      af::constant(float(target.elements()), 1));
                }
                fl::Dataset::BatchFunction fnc =
                    [targetpadVal](const std::vector<af::array>& arr) {
                      return fl::join(arr, targetpadVal, 1);
                    };
                auto newTargetsSizeBatch =
                    fl::makeBatch(newTargetsSize, nullptr);
                auto newTargetsBatch = fl::makeBatch(newTargets, fnc);
                meters.stats.add(
                    batch[kDurationIdx](maskedSamples), newTargetsSizeBatch);
                // TODO optimize masking early
                critArgs = {
                    output(af::span, af::span, maskedSamples, af::span, true),
                    fl::Variable(newTargetsBatch, false)};
                newUnsupDuration = batch[kDurationIdx](maskedSamples);
                if (isSeq2seqCrit) {
                  critArgs.push_back(fl::Variable(newUnsupDuration, false));
                  critArgs.push_back(fl::Variable(newTargetsSizeBatch, false));
                }
              } else {
                LOG(INFO)
                    << "Skip unsupervised part of data as PL are not available yet";
              }
            }
          }
          float r = critArgs.size() > 0;
          af::array doUpdateArr = af::array(1, &r);
          af::sync();
          if (FLAGS_enable_distributed) {
            fl::allReduce(doUpdateArr);
          }
          if (af::sum<int>(doUpdateArr) < fl::getWorldSize()) {
            doUpdate = false;
            break;
          }
          meters.critfwdtimer.resume();
          fl::Variable loss;
          if (!isSupBatch && FLAGS_slimIPL_use_soft) {
            af::print("target", critArgs[1].array());
            af::print("pred", critArgs[0].array());
            loss = FLAGS_slimIPL_soft_scale *
                fl::negate(fl::mean(
                    fl::sum(
                        fl::softmax(critArgs[1].as(f32), 0) *
                            fl::logSoftmax(critArgs[0].as(f32), 0),
                        {0}),
                    {1, 2, 3}));
          } else {
            loss = crit->forward(critArgs).front();
          }
          af::sync();
          meters.fwdtimer.stopAndIncUnit();
          meters.critfwdtimer.stopAndIncUnit();

          if (FLAGS_fl_amp_use_mixed_precision) {
            ++scaleCounter;
            loss = loss * scaleFactor;
          }

          if (af::anyTrue<bool>(af::isNaN(loss.array())) ||
              af::anyTrue<bool>(af::isInf(loss.array()))) {
            if (af::anyTrue<bool>(af::isInf(critArgs[0].array()))) {
              LOG(INFO) << "input to crit has Inf values. Samples - "
                        << join(",", readSampleIds(batch[kSampleIdx]));
            }
            if (af::anyTrue<bool>(af::isNaN(critArgs[0].array()))) {
              LOG(INFO) << "input to crit has NaN values. Samples - "
                        << join(",", readSampleIds(batch[kSampleIdx]));
            }
            LOG(FATAL) << "Loss has NaN values. Samples - "
                       << join(",", readSampleIds(batch[kSampleIdx]));
          }
          if (hasher(join(",", readSampleIds(batch[kSampleIdx]))) % 100 <=
              FLAGS_pcttraineval) {
            if (isSupBatch) {
              evalOutput(
                  critArgs[0].array(),
                  critArgs[1].array(),
                  batch[kDurationIdx],
                  meters.train);
            } else {
              evalOutput(
                  critArgs[0].array(),
                  critArgs[1].array(),
                  newUnsupDuration,
                  meters.trainUnsup);
            }
          }

          // backward
          meters.bwdtimer.resume();
          netopt->zeroGrad();
          critopt->zeroGrad();
          loss.backward();
          if (reducer) {
            for (auto& p : ntwrk->params()) {
              if (!p.isGradAvailable()) {
                p.addGrad(fl::constant(0.0, p.dims(), p.type(), false));
              }
              reducer->add(p.grad());
            }
            for (auto& p : crit->params()) {
              if (!p.isGradAvailable()) {
                p.addGrad(fl::constant(0.0, p.dims(), p.type(), false));
              }
              reducer->add(p.grad());
            }
            reducer->finalize();
          }
          af::sync();
          meters.bwdtimer.stopAndIncUnit();

          // optimizer
          meters.optimtimer.resume();

          // scale down gradients by batchsize
          af::array totalBatchSizeArr = af::constant(loss.dims(0), 1, f32);
          if (reducer) {
            fl::allReduce(totalBatchSizeArr);
          }
          float totalBatchSize = totalBatchSizeArr.scalar<float>();
          for (const auto& p : ntwrk->params()) {
            if (!p.isGradAvailable()) {
              continue;
            }
            p.grad() = p.grad() / (totalBatchSize * scaleFactor);
            if (FLAGS_fl_amp_use_mixed_precision) {
              if (af::anyTrue<bool>(af::isNaN(p.grad().array())) ||
                  af::anyTrue<bool>(af::isInf(p.grad().array()))) {
                if (scaleFactor >= fl::kAmpMinimumScaleFactorValue) {
                  scaleFactor = scaleFactor / 2.0f;
                  FL_VLOG(2) << "AMP: Scale factor decreased. New value:\t"
                             << scaleFactor;
                  retrySample = true;
                }
                scaleCounter = 1;
                break;
              }
            }
          }
          if (retrySample) {
            LOG(INFO) << "Retry amp sample " << scaleFactor;
            meters.optimtimer.stop();
            continue;
          }

          if (isSupBatch) {
            meters.train.loss.add((loss / scaleFactor).array());
          } else {
            meters.trainUnsup.loss.add((loss / scaleFactor).array());
          }

          for (const auto& p : crit->params()) {
            if (!p.isGradAvailable()) {
              continue;
            }
            p.grad() = p.grad() / (totalBatchSize * scaleFactor);
          }
        } while (retrySample);
        for (int index = 0; index < plTextArrayPreCacheToSave.size(); index++) {
          plCache[samplesIndices[index]] = plTextArrayPreCacheToSave[index];
        }
        if (doUpdate) {
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

          // update scale factor
          if (FLAGS_fl_amp_use_mixed_precision &&
              scaleFactor < kMaxScaleFactor) {
            if (scaleCounter % kScaleFactorUpdateInterval == 0) {
              scaleFactor *= 2;
              FL_VLOG(2) << "AMP: Scale factor doubled. New value:\t"
                         << scaleFactor;
            } else {
              scaleFactor += 2;
              FL_VLOG(3) << "AMP: Scale factor incremented. New value\t"
                         << scaleFactor;
            }
          }
        } else {
          LOG(INFO) << "Skip update step as unsup data has no label "
                    << curBatch;
        }
        // update EMA model
        if (FLAGS_slimIPL_ema) {
          for (int i = 0; i < ntwrkEMA->params().size(); ++i) {
            af::array newParam =
                ntwrkEMA->param(i).array() * FLAGS_slimIPL_ema_decay +
                ntwrk->param(i).array() * (1 - FLAGS_slimIPL_ema_decay);
            newParam.eval();
            ntwrkEMA->setParams(fl::Variable(newParam, false), i);
          }
        }
        if (!isSupBatch) {
          if (FLAGS_slimIPL_type == "cache") {
            auto plTextArray = predictPL(batch);
            auto samplesIndices = readSampleIds(batch[kSampleIdx]);
            for (int index = 0; index < plTextArray.size(); index++) {
              plCache[samplesIndices[index]] = plTextArray[index];
            }
          }
        }
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
        networkEMA,
        linseg,
        trainds,
        nullptr,
        linNetoptim,
        linCritoptim,
        initLinNetlr,
        initLinCritlr,
        false /* clampCrit */,
        FLAGS_linseg - startUpdate);

    startUpdate = FLAGS_linseg;
    FL_LOG_MASTER(INFO) << "Finished LinSeg";
  }

  auto s2s = std::dynamic_pointer_cast<Seq2SeqCriterion>(criterion);
  auto trde = std::dynamic_pointer_cast<TransformerCriterion>(criterion);
  if (FLAGS_pretrainWindow - startUpdate > 0) {
    if (!s2s && !trde) {
      LOG(FATAL) << "Window pretraining only allowed for seq2seq.";
    }
    train(
        network,
        networkEMA,
        criterion,
        trainds,
        nullptr,
        netoptim,
        critoptim,
        FLAGS_lr,
        FLAGS_lrcrit,
        true,
        FLAGS_pretrainWindow);
    startUpdate = FLAGS_pretrainWindow;
    FL_LOG_MASTER(INFO) << "Finished window pretraining.";
  }
  if (s2s) {
    s2s->clearWindow();
  } else if (trde) {
    trde->clearWindow();
  }

  if (FLAGS_slimIPL_start - startUpdate > 0) {
    train(
        network,
        networkEMA,
        criterion,
        trainds,
        nullptr,
        netoptim,
        critoptim,
        FLAGS_lr,
        FLAGS_lrcrit,
        true /* clampCrit */,
        FLAGS_slimIPL_start);
    startUpdate = FLAGS_slimIPL_start;
    FL_LOG_MASTER(INFO) << "Finished supervised only pretraining.";
  }

  train(
      network,
      networkEMA,
      criterion,
      trainds,
      unsupTrainds,
      netoptim,
      critoptim,
      FLAGS_lr,
      FLAGS_lrcrit,
      true /* clampCrit */,
      FLAGS_iter);

  FL_LOG_MASTER(INFO) << "Finished training";
  return 0;
}
