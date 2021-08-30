/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>
#include "flashlight/fl/flashlight.h"

#include "flashlight/app/asr/common/Defines.h"
#include "flashlight/app/asr/common/Flags.h"
#include "flashlight/app/asr/criterion/criterion.h"
#include "flashlight/app/asr/data/FeatureTransforms.h"
#include "flashlight/app/asr/data/Utils.h"
#include "flashlight/app/asr/decoder/ConvLmModule.h"
#include "flashlight/app/asr/decoder/DecodeUtils.h"
#include "flashlight/app/asr/decoder/Defines.h"
#include "flashlight/app/asr/decoder/TranscriptionUtils.h"
#include "flashlight/app/asr/runtime/runtime.h"
#include "flashlight/ext/common/SequentialBuilder.h"
#include "flashlight/ext/common/Serializer.h"
#include "flashlight/lib/common/ProducerConsumerQueue.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeDecoder.h"
#include "flashlight/lib/text/decoder/LexiconFreeSeq2SeqDecoder.h"
#include "flashlight/lib/text/decoder/LexiconSeq2SeqDecoder.h"
#include "flashlight/lib/text/decoder/lm/ConvLM.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/decoder/lm/ZeroLM.h"

#include "CPCCriterion.h"
#include "SequentialBuilder.h"

DECLARE_string(criterion2);
DEFINE_string(criterion2, "ctc", "Criterion for supervised task");

using fl::ext::afToVector;
using fl::ext::Serializer;
using fl::lib::join;
using fl::lib::pathsConcat;
using fl::lib::text::CriterionType;
using fl::lib::text::kUnkToken;
using fl::lib::text::SmearingMode;

using namespace fl::app::asr;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: Please refer to https://git.io/JvJuR");
  if (argc <= 1) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  auto flagsfile = FLAGS_flagsfile;
  if (!flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << flagsfile;
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
    // Re-parse command line flags to override values in the flag file.
    gflags::ParseCommandLineFlags(&argc, &argv, false);
  }

  if (!FLAGS_fl_log_level.empty()) {
    fl::Logging::setMaxLoggingLevel(fl::logLevelValue(FLAGS_fl_log_level));
  }
  fl::VerboseLogging::setMaxLoggingLevel(FLAGS_fl_vlog_level);

  /* ===================== Create Network ===================== */
  if (FLAGS_emission_dir.empty() && FLAGS_am.empty()) {
    LOG(FATAL) << "Both flags are empty: `-emission_dir` and `-am`";
  }

  std::shared_ptr<fl::Sequential> network;
  std::shared_ptr<SequenceCriterion> _criterion;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  std::string version;

  /* Using acoustic model */
  if (!FLAGS_am.empty()) {
    LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
    af::setDevice(0);
    Serializer::load(FLAGS_am, version, cfg, network, _criterion, criterion);
    network->eval();
    if (version != FL_APP_ASR_VERSION) {
      LOG(WARNING) << "[Network] Model version " << version
                   << " and code version " << FL_APP_ASR_VERSION;
    }
    LOG(INFO) << "[Network] " << network->prettyString();
    if (criterion) {
      criterion->eval();
      LOG(INFO) << "[Criterion] " << criterion->prettyString();
    }
    LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
    }
    LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
  }

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
    // Re-parse command line flags to override values in the flag file.
    gflags::ParseCommandLineFlags(&argc, &argv, false);
  }

  // Only Copy any values from deprecated flags to new flags when deprecated
  // flags are present and corresponding new flags aren't
  handleDeprecatedFlags();

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = FLAGS_tokens;
  if (dictPath.empty() || !fl::lib::fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
  }
  fl::lib::text::Dictionary tokenDict(dictPath);
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
    tokenDict.addEntry(fl::app::asr::kEosToken);
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
  } else {
    if (FLAGS_uselexicon || FLAGS_decodertype == "wrd") {
      LOG(FATAL) << "For lexicon-based beam-search decoder "
                 << "lexicon shouldn't be empty";
    }
  }

  /* =============== Prepare Sharable Decoder Components ============== */
  // Prepare counters
  std::vector<double> sliceWrdDst(FLAGS_nthread_decoder);
  std::vector<double> sliceTknDst(FLAGS_nthread_decoder);
  std::vector<int> sliceNumWords(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumTokens(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumSamples(FLAGS_nthread_decoder, 0);
  std::vector<double> sliceTime(FLAGS_nthread_decoder, 0);

  // Prepare criterion
  CriterionType criterionType = CriterionType::ASG;
  if (FLAGS_criterion2 == kCtcCriterion) {
    criterionType = CriterionType::CTC;
  } else if (
      FLAGS_criterion == kSeq2SeqRNNCriterion ||
      FLAGS_criterion == kSeq2SeqTransformerCriterion) {
    criterionType = CriterionType::S2S;
  } else if (FLAGS_criterion2 != kAsgCriterion) {
    LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion2;
  }

  std::vector<float> transition;
  if (FLAGS_criterion2 == kAsgCriterion) {
    transition = afToVector<float>(criterion->param(0).array());
  }

  // Prepare log writer
  std::mutex hypMutex, refMutex, logMutex;
  std::ofstream hypStream, refStream, logStream;
  if (!FLAGS_sclite.empty()) {
    auto fileName = cleanFilepath(FLAGS_test);
    auto hypPath = pathsConcat(FLAGS_sclite, fileName + ".hyp");
    auto refPath = pathsConcat(FLAGS_sclite, fileName + ".ref");
    auto logPath = pathsConcat(FLAGS_sclite, fileName + ".log");
    hypStream.open(hypPath);
    refStream.open(refPath);
    logStream.open(logPath);
    if (!hypStream.is_open() || !hypStream.good()) {
      LOG(FATAL) << "Error opening hypothesis file: " << hypPath;
    }
    if (!refStream.is_open() || !refStream.good()) {
      LOG(FATAL) << "Error opening reference file: " << refPath;
    }
    if (!logStream.is_open() || !logStream.good()) {
      LOG(FATAL) << "Error opening log file: " << logPath;
    }
  }

  auto writeHyp = [&hypMutex, &hypStream](const std::string& hypStr) {
    std::lock_guard<std::mutex> lock(hypMutex);
    hypStream << hypStr;
  };
  auto writeRef = [&refMutex, &refStream](const std::string& refStr) {
    std::lock_guard<std::mutex> lock(refMutex);
    refStream << refStr;
  };
  auto writeLog = [&logMutex, &logStream](const std::string& logStr) {
    std::lock_guard<std::mutex> lock(logMutex);
    logStream << logStr;
  };

  // Build Language Model
  int unkWordIdx = -1;

  fl::lib::text::Dictionary usrDict = tokenDict;
  if (!FLAGS_lm.empty() && FLAGS_decodertype == "wrd") {
    usrDict = wordDict;
    unkWordIdx = wordDict.getIndex(kUnkToken);
  }

  std::shared_ptr<fl::lib::text::LM> lm =
      std::make_shared<fl::lib::text::ZeroLM>();
  if (!FLAGS_lm.empty()) {
    if (FLAGS_lmtype == "kenlm") {
      lm = std::make_shared<fl::lib::text::KenLM>(FLAGS_lm, usrDict);
      if (!lm) {
        LOG(FATAL) << "[LM constructing] Failed to load LM: " << FLAGS_lm;
      }
    } else if (FLAGS_lmtype == "convlm") {
      af::setDevice(0);
      LOG(INFO) << "[ConvLM]: Loading LM from " << FLAGS_lm;
      std::shared_ptr<fl::Module> convLmModel;
      std::string convlmVersion;
      Serializer::load(FLAGS_lm, convlmVersion, convLmModel);
      if (convlmVersion != FL_APP_ASR_VERSION) {
        LOG(WARNING) << "[Convlm] Model version " << convlmVersion
                     << " and code version " << FL_APP_ASR_VERSION;
      }
      convLmModel->eval();

      auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
      lm = std::make_shared<fl::lib::text::ConvLM>(
          getConvLmScoreFunc,
          FLAGS_lm_vocab,
          usrDict,
          FLAGS_lm_memory,
          FLAGS_beamsize);
    } else {
      LOG(FATAL) << "[LM constructing] Invalid LM Type: " << FLAGS_lmtype;
    }
  }
  LOG(INFO) << "[Decoder] LM constructed.";

  // Build Trie
  int blankIdx =
      FLAGS_criterion2 == kCtcCriterion ? tokenDict.getIndex(kBlankToken) : -1;
  int silIdx = -1;
  if (FLAGS_wordseparator != "") {
    silIdx = tokenDict.getIndex(FLAGS_wordseparator);
  }
  std::shared_ptr<fl::lib::text::Trie> trie = buildTrie(
      FLAGS_decodertype,
      FLAGS_uselexicon,
      lm,
      FLAGS_smearing,
      tokenDict,
      lexicon,
      wordDict,
      silIdx,
      FLAGS_replabel);
  LOG(INFO) << "[Decoder] Trie smeared.\n";

  /* ===================== Create Dataset ===================== */
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
  FeatureType featType =
      getFeatureType(FLAGS_features_type, FLAGS_channels, featParams).second;

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

  auto inputTransform = inputFeatures(
      featParams,
      featType,
      {FLAGS_localnrmlleftctx, FLAGS_localnrmlrightctx},
      /*sfxConf=*/{});
  auto targetTransform = targetFeatures(tokenDict, lexicon, targetGenConfig);
  auto wordTransform = wordFeatures(wordDict);
  int targetpadVal = isSeq2seqCrit
      ? tokenDict.getIndex(fl::lib::text::kPadToken)
      : kTargetPadValue;
  int wordpadVal = wordDict.getIndex(kUnkToken);

  std::vector<std::string> testSplits = fl::lib::split(",", FLAGS_test, true);
  auto ds = createDataset(
      testSplits,
      FLAGS_datadir,
      1 /* batchsize */,
      inputTransform,
      targetTransform,
      wordTransform,
      std::make_tuple(0, targetpadVal, wordpadVal),
      0 /* worldrank */,
      1 /* worldsize */);

  int nSamples = ds->size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }
  LOG(INFO) << "[Dataset] Dataset loaded, with " << nSamples << " samples.";

  /* ===================== AM Forwarding ===================== */
  using EmissionQueue = fl::lib::ProducerConsumerQueue<EmissionTargetPair>;
  EmissionQueue emissionQueue(FLAGS_emission_queue_size);

  auto runAmForward = [&network,
                       &criterion,
                       &nSamples,
                       &ds,
                       &tokenDict,
                       &wordDict,
                       &emissionQueue,
                       &isSeq2seqCrit](int tid) {
    // Initialize AM
    af::setDevice(tid);
    std::shared_ptr<fl::Sequential> localNetwork = network;
    std::shared_ptr<SequenceCriterion> localCriterion = criterion;
    std::shared_ptr<SequenceCriterion> _localCriterion;
    if (tid != 0) {
      std::unordered_map<std::string, std::string> dummyCfg;
      std::string dummyVersion;
      Serializer::load(
          FLAGS_am,
          dummyVersion,
          dummyCfg,
          localNetwork,
          _localCriterion,
          localNetwork);
      localNetwork->eval();
      localCriterion->eval();
    }

    std::vector<int64_t> selectedIds;
    for (int64_t i = tid; i < nSamples; i += FLAGS_nthread_decoder_am_forward) {
      selectedIds.emplace_back(i);
    }
    std::shared_ptr<fl::Dataset> localDs =
        std::make_shared<fl::ResampleDataset>(ds, selectedIds);
    localDs = std::make_shared<fl::PrefetchDataset>(
        localDs, FLAGS_nthread, FLAGS_nthread);

    for (auto& sample : *localDs) {
      auto sampleId = readSampleIds(sample[kSampleIdx]).front();

      /* 2. Load Targets */
      TargetUnit targetUnit;
      auto tokenTarget = afToVector<int>(sample[kTargetIdx]);
      auto wordTarget = afToVector<int>(sample[kWordIdx]);
      // TODO: we will reform the dataset so that the loaded word
      // targets are strings already
      std::vector<std::string> wordTargetStr;
      if (FLAGS_uselexicon) {
        wordTargetStr = wrdIdx2Wrd(wordTarget, wordDict);
      } else {
        auto letterTarget = tknTarget2Ltr(
            tokenTarget,
            tokenDict,
            FLAGS_criterion2,
            FLAGS_surround,
            isSeq2seqCrit,
            FLAGS_replabel,
            FLAGS_usewordpiece,
            FLAGS_wordseparator);
        wordTargetStr = tkn2Wrd(letterTarget, FLAGS_wordseparator);
      }

      targetUnit.wordTargetStr = wordTargetStr;
      targetUnit.tokenTarget = tokenTarget;

      /* 3. Load Emissions */
      EmissionUnit emissionUnit;
      if (FLAGS_emission_dir.empty()) {
        int idx = 0;
        auto enc_out = localNetwork->module(idx++)
                           ->forward({fl::input(sample[kInputIdx])})
                           .front();
        enc_out = localNetwork->module(idx++)->forward({enc_out}).front();
        enc_out = localNetwork->module(idx++)->forward({enc_out}).front();
        enc_out = w2l::cpc::forwardSequentialModuleWithPadMask(
            enc_out, localNetwork->module(idx++), sample[kDurationIdx]);
        auto rawEmission =
            localNetwork->module(idx)->forward({enc_out}).front();
        emissionUnit = EmissionUnit(
            afToVector<float>(rawEmission),
            sampleId,
            rawEmission.dims(1),
            rawEmission.dims(0));
      } else {
        auto cleanTestPath = cleanFilepath(FLAGS_test);
        std::string emissionDir =
            pathsConcat(FLAGS_emission_dir, cleanTestPath);
        std::string savePath = pathsConcat(emissionDir, sampleId + ".bin");
        std::string eVersion;
        Serializer::load(savePath, eVersion, emissionUnit);
      }

      emissionQueue.add({emissionUnit, targetUnit});
    }

    localNetwork.reset(); // AM is only used in running forward pass. So we will
    // free the space of it on GPU or memory.
    // localNetwork.use_count() will be 0 after this call.

    af::deviceGC(); // Explicitly call the Garbage collector.
  };

  /* ===================== Decode ===================== */
  auto runDecoder = [&criterion,
                     &lm,
                     &trie,
                     &silIdx,
                     &blankIdx,
                     &unkWordIdx,
                     &criterionType,
                     &transition,
                     &usrDict,
                     &tokenDict,
                     &wordDict,
                     &emissionQueue,
                     &writeHyp,
                     &writeRef,
                     &writeLog,
                     &sliceWrdDst,
                     &sliceTknDst,
                     &sliceNumWords,
                     &sliceNumTokens,
                     &sliceNumSamples,
                     &sliceTime,
                     &isSeq2seqCrit](int tid) {
    /* 1. Prepare GPU-dependent resources */
    // Note: These 2 GPU-dependent models should be placed on different
    // cards
    // for different threads and nthread_decoder should not be greater
    // than
    // the number of GPUs.
    std::shared_ptr<SequenceCriterion> localCriterion = criterion;
    std::shared_ptr<fl::lib::text::LM> localLm = lm;
    if (FLAGS_lmtype == "convlm" || criterionType == CriterionType::S2S) {
      if (tid >= af::getDeviceCount()) {
        LOG(FATAL)
            << "FLAGS_nthread_decoder exceeds the number of visible GPUs";
      }
      af::setDevice(tid);
    }

    // Make a copy for non-main threads.
    if (tid != 0) {
      if (FLAGS_lmtype == "convlm") {
        LOG(INFO) << "[ConvLM]: Loading LM from " << FLAGS_lm;
        std::shared_ptr<fl::Module> convLmModel;
        std::string convlmVersion;
        Serializer::load(FLAGS_lm, convlmVersion, convLmModel);
        convLmModel->eval();

        auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
        localLm = std::make_shared<fl::lib::text::ConvLM>(
            getConvLmScoreFunc,
            FLAGS_lm_vocab,
            usrDict,
            FLAGS_lm_memory,
            FLAGS_beamsize);
      }

      if (criterionType == CriterionType::S2S) {
        std::shared_ptr<fl::Module> dummyNetwork;
        std::unordered_map<std::string, std::string> dummyCfg;
        Serializer::load(FLAGS_am, dummyCfg, dummyNetwork, localCriterion);
        localCriterion->eval();
      }
    }

    /* 2. Build Decoder */
    std::unique_ptr<fl::lib::text::Decoder> decoder;
    if (FLAGS_decodertype != "wrd" && FLAGS_decodertype != "tkn") {
      LOG(FATAL) << "Unsupported decoder type: " << FLAGS_decodertype;
    }

    if (criterionType == CriterionType::S2S) {
      auto amUpdateFunc = FLAGS_criterion == kSeq2SeqRNNCriterion
          ? buildSeq2SeqRnnAmUpdateFunction(
                localCriterion,
                FLAGS_decoderattnround,
                FLAGS_beamsize,
                FLAGS_attentionthreshold,
                FLAGS_smoothingtemperature)
          : buildSeq2SeqTransformerAmUpdateFunction(
                localCriterion,
                FLAGS_beamsize,
                FLAGS_attentionthreshold,
                FLAGS_smoothingtemperature);
      int eosIdx = tokenDict.getIndex(fl::app::asr::kEosToken);

      if (FLAGS_decodertype == "wrd" || FLAGS_uselexicon) {
        decoder.reset(new fl::lib::text::LexiconSeq2SeqDecoder(
            {
                .beamSize = FLAGS_beamsize,
                .beamSizeToken = FLAGS_beamsizetoken,
                .beamThreshold = FLAGS_beamthreshold,
                .lmWeight = FLAGS_lmweight,
                .wordScore = FLAGS_wordscore,
                .eosScore = FLAGS_eosscore,
                .logAdd = FLAGS_logadd,
            },
            trie,
            localLm,
            eosIdx,
            amUpdateFunc,
            FLAGS_maxdecoderoutputlen,
            FLAGS_decodertype == "tkn"));
        LOG(INFO) << "[Decoder] LexiconSeq2Seq decoder with "
                  << FLAGS_decodertype << "-LM loaded in thread: " << tid;
      } else {
        decoder.reset(new fl::lib::text::LexiconFreeSeq2SeqDecoder(
            {
                .beamSize = FLAGS_beamsize,
                .beamSizeToken = FLAGS_beamsizetoken,
                .beamThreshold = FLAGS_beamthreshold,
                .lmWeight = FLAGS_lmweight,
                .eosScore = FLAGS_eosscore,
                .logAdd = FLAGS_logadd,
            },
            localLm,
            eosIdx,
            amUpdateFunc,
            FLAGS_maxdecoderoutputlen));
        LOG(INFO)
            << "[Decoder] LexiconFreeSeq2Seq decoder with token-LM loaded in thread: "
            << tid;
      }
    } else {
      if (FLAGS_decodertype == "wrd" || FLAGS_uselexicon) {
        decoder.reset(new fl::lib::text::LexiconDecoder(
            {.beamSize = FLAGS_beamsize,
             .beamSizeToken = FLAGS_beamsizetoken,
             .beamThreshold = FLAGS_beamthreshold,
             .lmWeight = FLAGS_lmweight,
             .wordScore = FLAGS_wordscore,
             .unkScore = FLAGS_unkscore,
             .silScore = FLAGS_silscore,
             .logAdd = FLAGS_logadd,
             .criterionType = criterionType},
            trie,
            localLm,
            silIdx,
            blankIdx,
            unkWordIdx,
            transition,
            FLAGS_decodertype == "tkn"));
        LOG(INFO) << "[Decoder] Lexicon decoder with " << FLAGS_decodertype
                  << "-LM loaded in thread: " << tid;
      } else {
        decoder.reset(new fl::lib::text::LexiconFreeDecoder(
            {.beamSize = FLAGS_beamsize,
             .beamSizeToken = FLAGS_beamsizetoken,
             .beamThreshold = FLAGS_beamthreshold,
             .lmWeight = FLAGS_lmweight,
             .silScore = FLAGS_silscore,
             .logAdd = FLAGS_logadd,
             .criterionType = criterionType},
            localLm,
            silIdx,
            blankIdx,
            transition));
        LOG(INFO)
            << "[Decoder] Lexicon-free decoder with token-LM loaded in thread: "
            << tid;
      }
    }
    /* 3. Get data and run decoder */
    TestMeters meters;
    EmissionTargetPair emissionTargetPair;
    while (emissionQueue.get(emissionTargetPair)) {
      const auto& emissionUnit = emissionTargetPair.first;
      const auto& targetUnit = emissionTargetPair.second;

      const auto& nFrames = emissionUnit.nFrames;
      const auto& nTokens = emissionUnit.nTokens;
      const auto& emission = emissionUnit.emission;
      const auto& sampleId = emissionUnit.sampleId;
      const auto& wordTarget = targetUnit.wordTargetStr;
      const auto& tokenTarget = targetUnit.tokenTarget;
      // DecodeResult
      meters.timer.reset();
      meters.timer.resume();
      const auto& results = decoder->decode(emission.data(), nFrames, nTokens);
      meters.timer.stop();

      int nTopHyps = FLAGS_isbeamdump ? results.size() : 1;
      for (int i = 0; i < nTopHyps; i++) {
        // Cleanup predictions
        auto rawWordPrediction = results[i].words;
        auto rawTokenPrediction = results[i].tokens;

        auto letterTarget = tknTarget2Ltr(
            tokenTarget,
            tokenDict,
            FLAGS_criterion2,
            FLAGS_surround,
            isSeq2seqCrit,
            FLAGS_replabel,
            FLAGS_usewordpiece,
            FLAGS_wordseparator);
        auto letterPrediction = tknPrediction2Ltr(
            rawTokenPrediction,
            tokenDict,
            FLAGS_criterion2,
            FLAGS_surround,
            isSeq2seqCrit,
            FLAGS_replabel,
            FLAGS_usewordpiece,
            FLAGS_wordseparator);
        std::vector<std::string> wordPrediction;
        if (FLAGS_uselexicon) {
          rawWordPrediction =
              validateIdx(rawWordPrediction, wordDict.getIndex(kUnkToken));
          wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);
        } else {
          wordPrediction = tkn2Wrd(letterPrediction, FLAGS_wordseparator);
        }
        auto wordTargetStr = join(" ", wordTarget);
        auto wordPredictionStr = join(" ", wordPrediction);
        // Normal decoding and computing WER
        if (!FLAGS_isbeamdump) {
          meters.wrdDstSlice.add(wordPrediction, wordTarget);
          meters.tknDstSlice.add(letterPrediction, letterTarget);

          if (!FLAGS_sclite.empty()) {
            std::string suffix = " (" + sampleId + ")\n";
            writeHyp(wordPredictionStr + suffix);
            writeRef(wordTargetStr + suffix);
          }

          if (FLAGS_show) {
            meters.wrdDst.reset();
            meters.tknDst.reset();
            meters.wrdDst.add(wordPrediction, wordTarget);
            meters.tknDst.add(letterPrediction, letterTarget);

            std::stringstream buffer;
            buffer << "|T|: " << wordTargetStr << std::endl;
            buffer << "|P|: " << wordPredictionStr << std::endl;
            if (FLAGS_showletters) {
              buffer << "|t|: " << join(" ", letterTarget) << std::endl;
              buffer << "|p|: " << join(" ", letterPrediction) << std::endl;
            }
            buffer << "[sample: " << sampleId
                   << ", WER: " << meters.wrdDst.errorRate()[0]
                   << "\%, TER: " << meters.tknDst.errorRate()[0]
                   << "\%, slice WER: " << meters.wrdDstSlice.errorRate()[0]
                   << "\%, slice TER: " << meters.tknDstSlice.errorRate()[0]
                   << "\%, decoded samples (thread " << tid
                   << "): " << sliceNumSamples[tid] + 1 << "]" << std::endl;

            std::cout << buffer.str();

            if (!FLAGS_sclite.empty()) {
              writeLog(buffer.str());
            }
          }

          // Update conters
          sliceNumWords[tid] += wordTarget.size();
          sliceNumTokens[tid] += letterTarget.size();
          sliceTime[tid] += meters.timer.value();
          sliceNumSamples[tid] += 1;
        }
        // Beam Dump
        else {
          meters.wrdDst.reset();
          meters.wrdDst.add(wordPrediction, wordTarget);
          auto wer = meters.wrdDst.errorRate()[0];

          if (FLAGS_sclite.empty()) {
            LOG(FATAL) << "FLAGS_sclite is empty, nowhere to dump the beam.";
          }

          auto score = results[i].score;
          auto amScore = results[i].amScore;
          auto lmScore = results[i].lmScore;
          auto outString = sampleId + " | " + std::to_string(score) + " | " +
              std::to_string(amScore) + " | " + std::to_string(lmScore) +
              " | " + std::to_string(wer) + " | " + wordPredictionStr + "\n";
          writeHyp(outString);
        }
      }
    }
    sliceWrdDst[tid] = meters.wrdDstSlice.value()[0];
    sliceTknDst[tid] = meters.tknDstSlice.value()[0];
  };

  /* ===================== Spread threades ===================== */
  if (FLAGS_nthread_decoder_am_forward <= 0) {
    LOG(FATAL) << "FLAGS_nthread_decoder_am_forward ("
               << FLAGS_nthread_decoder_am_forward << ") need to be positive ";
  }
  if (FLAGS_nthread_decoder <= 0) {
    LOG(FATAL) << "FLAGS_nthread_decoder (" << FLAGS_nthread_decoder
               << ") need to be positive ";
  }

  auto startThreadsAndJoin = [&runAmForward, &runDecoder, &emissionQueue](
                                 int nAmThreads, int nDecoderThreads) {
    // TODO possibly try catch for futures to proper logging of all errors
    // https://github.com/facebookresearch/gtn/blob/master/gtn/parallel/parallel_map.h#L154

    // We have to run AM forwarding and decoding in sequential to avoid GPU
    // OOM with two large neural nets.
    if (FLAGS_lmtype == "convlm") {
      // 1. AM forwarding
      {
        std::vector<std::future<void>> futs(nAmThreads);
        fl::ThreadPool threadPool(nAmThreads);
        for (int i = 0; i < nAmThreads; i++) {
          futs[i] = threadPool.enqueue(runAmForward, i);
        }
        for (int i = 0; i < nAmThreads; i++) {
          futs[i].get();
        }
        emissionQueue.finishAdding();
      }
      // 2. Decoding
      {
        std::vector<std::future<void>> futs(nDecoderThreads);
        fl::ThreadPool threadPool(nDecoderThreads);
        for (int i = 0; i < nDecoderThreads; i++) {
          futs[i] = threadPool.enqueue(runDecoder, i);
        }
        for (int i = 0; i < nDecoderThreads; i++) {
          futs[i].get();
        }
      }
    }
    // Non-convLM decoding. AM forwarding and decoding can be run in parallel.
    else {
      std::vector<std::future<void>> futs(nAmThreads + nDecoderThreads);
      fl::ThreadPool threadPool(nAmThreads + nDecoderThreads);
      // AM forwarding threads
      for (int i = 0; i < nAmThreads; i++) {
        futs[i] = threadPool.enqueue(runAmForward, i);
      }
      // Decoding threads
      for (int i = 0; i < nDecoderThreads; i++) {
        futs[i + nAmThreads] = threadPool.enqueue(runDecoder, i);
      }

      for (int i = 0; i < nAmThreads; i++) {
        futs[i].get();
      }
      emissionQueue.finishAdding();
      for (int i = nAmThreads; i < nAmThreads + nDecoderThreads; i++) {
        futs[i].get();
      }
    }
  };
  auto timer = fl::TimeMeter();
  timer.resume();
  startThreadsAndJoin(FLAGS_nthread_decoder_am_forward, FLAGS_nthread_decoder);
  timer.stop();

  /* Compute statistics */
  int totalTokens = 0, totalWords = 0, totalSamples = 0;
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    totalTokens += sliceNumTokens[i];
    totalWords += sliceNumWords[i];
    totalSamples += sliceNumSamples[i];
  }
  double totalWer = 0, totalTkn = 0, totalTime = 0;
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    totalWer += sliceWrdDst[i];
    totalTkn += sliceTknDst[i];
    totalTime += sliceTime[i];
  }
  if (totalWer > 0 && totalWords == 0) {
    totalWer = std::numeric_limits<double>::infinity();
  } else {
    totalWer = totalWords > 0 ? totalWer / totalWords * 100. : 0.0;
  }
  if (totalTkn > 0 && totalTokens == 0) {
    totalTkn = std::numeric_limits<double>::infinity();
  } else {
    totalTkn = totalTokens > 0 ? totalTkn / totalTokens * 100. : 0.0;
  }
  std::stringstream buffer;
  buffer << "------\n";
  buffer << "[Decode " << FLAGS_test << " (" << totalSamples << " samples) in "
         << timer.value() << "s (actual decoding time " << std::setprecision(3)
         << totalTime / totalSamples
         << "s/sample) -- WER: " << std::setprecision(6) << totalWer
         << "\%, TER: " << totalTkn << "\%]" << std::endl;
  LOG(INFO) << buffer.str();
  if (!FLAGS_sclite.empty()) {
    writeLog(buffer.str());
    hypStream.close();
    refStream.close();
    logStream.close();
  }
  return 0;
}
