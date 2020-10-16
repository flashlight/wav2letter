/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "libraries/common/Dictionary.h"
#include "libraries/common/ProducerConsumerQueue.h"
#include "libraries/decoder/LexiconDecoder.h"
#include "libraries/decoder/LexiconFreeDecoder.h"
#include "libraries/decoder/LexiconFreeSeq2SeqDecoder.h"
#include "libraries/decoder/LexiconSeq2SeqDecoder.h"
#include "libraries/lm/ConvLM.h"
#include "libraries/lm/KenLM.h"
#include "libraries/lm/ZeroLM.h"
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

  /* ===================== Create Network ===================== */
  if (FLAGS_emission_dir.empty() && FLAGS_am.empty()) {
    LOG(FATAL) << "Both flags are empty: `-emission_dir` and `-am`";
  }

  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;

  /* Using acoustic model */
  if (!FLAGS_am.empty()) {
    LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
    af::setDevice(0);
    W2lSerializer::load(FLAGS_am, cfg, network, criterion);
    network->eval();
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
  w2l::handleDeprecatedFlags();

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::runtime_error("Invalid dictionary filepath specified.");
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

  /* =============== Prepare Sharable Decoder Components ============== */
  // Prepare counters
  std::vector<double> sliceWer(FLAGS_nthread_decoder);
  std::vector<double> sliceLer(FLAGS_nthread_decoder);
  std::vector<int> sliceNumWords(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumTokens(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumSamples(FLAGS_nthread_decoder, 0);
  std::vector<double> sliceTime(FLAGS_nthread_decoder, 0);

  // Prepare criterion
  CriterionType criterionType = CriterionType::ASG;
  if (FLAGS_criterion == kCtcCriterion) {
    criterionType = CriterionType::CTC;
  } else if (
      FLAGS_criterion == kSeq2SeqCriterion ||
      FLAGS_criterion == kTransformerCriterion) {
    criterionType = CriterionType::S2S;
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion;
  }

  std::vector<float> transition;
  if (FLAGS_criterion == kAsgCriterion) {
    transition = afToVector<float>(criterion->param(0).array());
  }

  // Prepare decoder options
  DecoderOptions decoderOpt(
      FLAGS_beamsize,
      FLAGS_beamsizetoken,
      FLAGS_beamthreshold,
      FLAGS_lmweight,
      FLAGS_wordscore,
      FLAGS_unkscore,
      FLAGS_silscore,
      FLAGS_eosscore,
      FLAGS_logadd,
      criterionType);

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

  Dictionary usrDict = tokenDict;
  if (!FLAGS_lm.empty() && FLAGS_decodertype == "wrd") {
    usrDict = wordDict;
    unkWordIdx = wordDict.getIndex(kUnkToken);
  }

  std::shared_ptr<LM> lm = std::make_shared<ZeroLM>();
  if (!FLAGS_lm.empty()) {
    if (FLAGS_lmtype == "kenlm") {
      lm = std::make_shared<KenLM>(FLAGS_lm, usrDict);
      if (!lm) {
        LOG(FATAL) << "[LM constructing] Failed to load LM: " << FLAGS_lm;
      }
    } else if (FLAGS_lmtype == "convlm") {
      af::setDevice(0);
      LOG(INFO) << "[ConvLM]: Loading LM from " << FLAGS_lm;
      std::shared_ptr<fl::Module> convLmModel;
      W2lSerializer::load(FLAGS_lm, convLmModel);
      convLmModel->eval();

      auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
      lm = std::make_shared<ConvLM>(
          getConvLmScoreFunc,
          FLAGS_lm_vocab,
          usrDict,
          FLAGS_lm_memory,
          FLAGS_beamsize);
    } else {
      LOG(FATAL) << "[LM constructing] Invalid LM Type: " << FLAGS_lmtype;
    }
  }
  LOG(INFO) << "[Decoder] LM constructed.\n";

  // Build Trie
  int blankIdx =
      FLAGS_criterion == kCtcCriterion ? tokenDict.getIndex(kBlankToken) : -1;
  int silIdx = -1;
  if (FLAGS_wordseparator != "") {
    silIdx = tokenDict.getIndex(FLAGS_wordseparator);
  }
  std::shared_ptr<Trie> trie = nullptr;
  if (FLAGS_decodertype == "wrd" || FLAGS_uselexicon) {
    trie = std::make_shared<Trie>(tokenDict.indexSize(), silIdx);
    auto startState = lm->start(false);

    for (auto& it : lexicon) {
      const std::string& word = it.first;
      int usrIdx = wordDict.getIndex(word);
      float score = -1;
      if (FLAGS_decodertype == "wrd") {
        LMStatePtr dummyState;
        std::tie(dummyState, score) = lm->score(startState, usrIdx);
      }
      for (auto& tokens : it.second) {
        auto tokensTensor = tkn2Idx(tokens, tokenDict, FLAGS_replabel);
        trie->insert(tokensTensor, usrIdx, score);
      }
    }
    LOG(INFO) << "[Decoder] Trie planted.\n";

    // Smearing
    SmearingMode smear_mode = SmearingMode::NONE;
    if (FLAGS_smearing == "logadd") {
      smear_mode = SmearingMode::LOGADD;
    } else if (FLAGS_smearing == "max") {
      smear_mode = SmearingMode::MAX;
    } else if (FLAGS_smearing != "none") {
      LOG(FATAL) << "[Decoder] Invalid smearing mode: " << FLAGS_smearing;
    }
    trie->smear(smear_mode);
    LOG(INFO) << "[Decoder] Trie smeared.\n";
  }

  /* ===================== AM Forwarding ===================== */
  using EmissionQueue = ProducerConsumerQueue<EmissionTargetPair>;
  EmissionQueue emissionQueue(FLAGS_emission_queue_size);

  // Load dataset
  auto ds = createDataset(
      FLAGS_test,
      dicts,
      lexicon,
      1 /* batchsize */,
      0 /* worldrank */,
      1 /* worldsize */);
  ds->shuffle(3);
  int nSamples = ds->size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }

  std::mutex dataReadMutex;
  int datasetGlobalSampleId = 0; // A gloabal index for data reading

  auto runAmForward = [&dataReadMutex,
                       &datasetGlobalSampleId,
                       &network,
                       &criterion,
                       &nSamples,
                       &ds,
                       &tokenDict,
                       &wordDict,
                       &emissionQueue](int tid) {
    // Initialize AM
    af::setDevice(tid);
    std::shared_ptr<fl::Module> localNetwork = network;
    std::shared_ptr<SequenceCriterion> localCriterion = criterion;
    if (tid != 0) {
      std::unordered_map<std::string, std::string> dummyCfg;
      W2lSerializer::load(FLAGS_am, dummyCfg, localNetwork, localCriterion);
      localNetwork->eval();
      localCriterion->eval();
    }

    while (datasetGlobalSampleId < nSamples) {
      /* 1. Get sample */
      int datasetLocalSampleId = -1;
      std::vector<af::array> sample;
      {
        std::lock_guard<std::mutex> lock(dataReadMutex);
        sample = ds->get(datasetGlobalSampleId);
        datasetLocalSampleId = datasetGlobalSampleId;
        datasetGlobalSampleId++;
      }
      auto sampleId = readSampleIds(sample[kSampleIdx]).front();

      /* 2. Load Targets */
      TargetUnit targetUnit;
      auto tokenTarget = afToVector<int>(sample[kTargetIdx]);
      auto wordTarget = afToVector<int>(sample[kWordIdx]);
      // TODO: we will reform the w2l dataset so that the loaded word targets
      // are strings already
      std::vector<std::string> wordTargetStr;
      if (FLAGS_uselexicon) {
        wordTargetStr = wrdIdx2Wrd(wordTarget, wordDict);
      } else {
        auto letterTarget = tknTarget2Ltr(tokenTarget, tokenDict);
        wordTargetStr = tkn2Wrd(letterTarget);
      }

      targetUnit.wordTargetStr = wordTargetStr;
      targetUnit.tokenTarget = tokenTarget;

      /* 3. Load Emissions */
      EmissionUnit emissionUnit;
      if (FLAGS_emission_dir.empty()) {
        auto rawEmission =
            localNetwork->forward({fl::input(sample[kInputIdx])}).front();
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
        W2lSerializer::load(savePath, emissionUnit);
      }

      emissionQueue.add({emissionUnit, targetUnit});
      if (datasetLocalSampleId == nSamples - 1) {
        emissionQueue.finishAdding();
      }
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
                     &decoderOpt,
                     &emissionQueue,
                     &writeHyp,
                     &writeRef,
                     &writeLog,
                     &sliceWer,
                     &sliceLer,
                     &sliceNumWords,
                     &sliceNumTokens,
                     &sliceNumSamples,
                     &sliceTime](int tid) {
    try {
      /* 1. Prepare GPU-dependent resources */
      // Note: These 2 GPU-dependent models should be placed on different cards
      // for different threads and nthread_decoder should not be greater than
      // the number of GPUs.
      std::shared_ptr<SequenceCriterion> localCriterion = criterion;
      std::shared_ptr<LM> localLm = lm;
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
          W2lSerializer::load(FLAGS_lm, convLmModel);
          convLmModel->eval();

          auto getConvLmScoreFunc = buildGetConvLmScoreFunction(convLmModel);
          localLm = std::make_shared<ConvLM>(
              getConvLmScoreFunc,
              FLAGS_lm_vocab,
              usrDict,
              FLAGS_lm_memory,
              FLAGS_beamsize);
        }

        if (criterionType == CriterionType::S2S) {
          std::shared_ptr<fl::Module> dummyNetwork;
          std::unordered_map<std::string, std::string> dummyCfg;
          W2lSerializer::load(FLAGS_am, dummyCfg, dummyNetwork, localCriterion);
          localCriterion->eval();
        }
      }

      /* 2. Build Decoder */
      std::unique_ptr<Decoder> decoder;
      if (criterionType == CriterionType::S2S) {
        auto amUpdateFunc = FLAGS_criterion == kSeq2SeqCriterion
            ? buildAmUpdateFunction(localCriterion)
            : buildTransformerAmUpdateFunction(localCriterion);
        int eosIdx = tokenDict.getIndex(kEosToken);

        if (FLAGS_decodertype == "wrd") {
          decoder.reset(new LexiconSeq2SeqDecoder(
              decoderOpt,
              trie,
              localLm,
              eosIdx,
              amUpdateFunc,
              FLAGS_maxdecoderoutputlen,
              false));
          LOG(INFO)
              << "[Decoder] LexiconSeq2Seq decoder with word-LM loaded in thread: "
              << tid;
        } else if (FLAGS_decodertype == "tkn") {
          if (FLAGS_uselexicon) {
            decoder.reset(new LexiconSeq2SeqDecoder(
                decoderOpt,
                trie,
                localLm,
                eosIdx,
                amUpdateFunc,
                FLAGS_maxdecoderoutputlen,
                true));
            LOG(INFO)
                << "[Decoder] LexiconSeq2Seq decoder with token-LM loaded in thread: "
                << tid;
          } else {
            decoder.reset(new LexiconFreeSeq2SeqDecoder(
                decoderOpt,
                localLm,
                eosIdx,
                amUpdateFunc,
                FLAGS_maxdecoderoutputlen));
            LOG(INFO)
                << "[Decoder] LexiconFreeSeq2Seq decoder with token-LM loaded in thread: "
                << tid;
          }
        } else {
          LOG(FATAL) << "Unsupported decoder type: " << FLAGS_decodertype;
        }
      } else {
        if (FLAGS_decodertype == "wrd") {
          decoder.reset(new LexiconDecoder(
              decoderOpt,
              trie,
              localLm,
              silIdx,
              blankIdx,
              unkWordIdx,
              transition,
              false));
          LOG(INFO)
              << "[Decoder] Lexicon decoder with word-LM loaded in thread: "
              << tid;
        } else if (FLAGS_decodertype == "tkn") {
          if (FLAGS_uselexicon) {
            decoder.reset(new LexiconDecoder(
                decoderOpt,
                trie,
                localLm,
                silIdx,
                blankIdx,
                unkWordIdx,
                transition,
                true));
            LOG(INFO)
                << "[Decoder] Lexicon decoder with token-LM loaded in thread: "
                << tid;
          } else {
            decoder.reset(new LexiconFreeDecoder(
                decoderOpt, localLm, silIdx, blankIdx, transition));
            LOG(INFO)
                << "[Decoder] Lexicon-free decoder with token-LM loaded in thread: "
                << tid;
          }
        } else {
          LOG(FATAL) << "Unsupported decoder type: " << FLAGS_decodertype;
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
        const auto& results =
            decoder->decode(emission.data(), nFrames, nTokens);
        meters.timer.stop();

        int nTopHyps = FLAGS_isbeamdump ? results.size() : 1;
        for (int i = 0; i < nTopHyps; i++) {
          // Cleanup predictions
          auto rawWordPrediction = results[i].words;
          auto rawTokenPrediction = results[i].tokens;

          auto letterTarget = tknTarget2Ltr(tokenTarget, tokenDict);
          auto letterPrediction =
              tknPrediction2Ltr(rawTokenPrediction, tokenDict);
          std::vector<std::string> wordPrediction;
          if (FLAGS_uselexicon) {
            rawWordPrediction =
                validateIdx(rawWordPrediction, wordDict.getIndex(kUnkToken));
            wordPrediction = wrdIdx2Wrd(rawWordPrediction, wordDict);
          } else {
            wordPrediction = tkn2Wrd(letterPrediction);
          }
          auto wordTargetStr = join(" ", wordTarget);
          auto wordPredictionStr = join(" ", wordPrediction);

          // Normal decoding and computing WER
          if (!FLAGS_isbeamdump) {
            meters.werSlice.add(wordPrediction, wordTarget);
            meters.lerSlice.add(letterPrediction, letterTarget);

            if (!FLAGS_sclite.empty()) {
              std::string suffix = " (" + sampleId + ")\n";
              writeHyp(wordPredictionStr + suffix);
              writeRef(wordTargetStr + suffix);
            }

            if (FLAGS_show) {
              meters.wer.reset();
              meters.ler.reset();
              meters.wer.add(wordPrediction, wordTarget);
              meters.ler.add(letterPrediction, letterTarget);

              std::stringstream buffer;
              buffer << "|T|: " << wordTargetStr << std::endl;
              buffer << "|P|: " << wordPredictionStr << std::endl;
              if (FLAGS_showletters) {
                buffer << "|t|: " << join(" ", letterTarget) << std::endl;
                buffer << "|p|: " << join(" ", letterPrediction) << std::endl;
              }
              buffer << "[sample: " << sampleId
                     << ", WER: " << meters.wer.value()[0]
                     << "\%, LER: " << meters.ler.value()[0]
                     << "\%, slice WER: " << meters.werSlice.value()[0]
                     << "\%, slice LER: " << meters.lerSlice.value()[0]
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
            meters.wer.reset();
            meters.wer.add(wordPrediction, wordTarget);
            auto wer = meters.wer.value()[0];

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
      sliceWer[tid] = meters.werSlice.value()[0];
      sliceLer[tid] = meters.lerSlice.value()[0];
    } catch (const std::exception& exc) {
      LOG(FATAL) << "Exception in thread " << tid << "\n" << exc.what();
    }
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

  auto startThreadsAndJoin = [&runAmForward, &runDecoder](
                                 int nAmThreads, int nDecoderThreads) {
    // We have to run AM forwarding and decoding in sequential to avoid GPU
    // OOM with two large neural nets.
    if (FLAGS_lmtype == "convlm") {
      // 1. AM forwarding
      {
        fl::ThreadPool threadPool(nAmThreads);
        for (int i = 0; i < nAmThreads; i++) {
          threadPool.enqueue(runAmForward, i);
        }
      }
      // 2. Decoding
      {
        fl::ThreadPool threadPool(nDecoderThreads);
        for (int i = 0; i < nDecoderThreads; i++) {
          threadPool.enqueue(runDecoder, i);
        }
      }
    }
    // Non-convLM decoding. AM forwarding and decoding can be run in parallel.
    else {
      fl::ThreadPool threadPool(nAmThreads + nDecoderThreads);
      // AM forwarding threads
      for (int i = 0; i < nAmThreads; i++) {
        threadPool.enqueue(runAmForward, i);
      }
      // Decoding threads
      for (int i = 0; i < nDecoderThreads; i++) {
        threadPool.enqueue(runDecoder, i);
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
  double totalWer = 0, totalLer = 0, totalTime = 0;
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    totalWer += sliceWer[i] * sliceNumWords[i] / totalWords;
    totalLer += sliceLer[i] * sliceNumTokens[i] / totalTokens;
    totalTime += sliceTime[i];
  }

  std::stringstream buffer;
  buffer << "------\n";
  buffer << "[Decode " << FLAGS_test << " (" << totalSamples << " samples) in "
         << timer.value() << "s (actual decoding time " << std::setprecision(3)
         << totalTime / totalSamples
         << "s/sample) -- WER: " << std::setprecision(6) << totalWer
         << ", LER: " << totalLer << "]" << std::endl;
  LOG(INFO) << buffer.str();
  if (!FLAGS_sclite.empty()) {
    writeLog(buffer.str());
    hypStream.close();
    refStream.close();
    logStream.close();
  }
  return 0;
}
