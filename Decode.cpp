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
#include "libraries/decoder/LexiconFreeDecoder.h"
#include "libraries/decoder/Seq2SeqDecoder.h"
#include "libraries/decoder/TokenLMDecoder.h"
#include "libraries/decoder/WordLMDecoder.h"
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
  gflags::SetUsageMessage("Usage: Please refer to https://git.io/fjVVq");
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

  EmissionSet emissionSet;
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;

  /* Using existing emissions */
  if (!FLAGS_emission_dir.empty()) {
    std::string cleanedTestPath = cleanFilepath(FLAGS_test);
    std::string loadPath =
        pathsConcat(FLAGS_emission_dir, cleanedTestPath + ".bin");
    LOG(INFO) << "[Serialization] Loading file: " << loadPath;
    W2lSerializer::load(loadPath, emissionSet);
    gflags::ReadFlagsFromString(emissionSet.gflags, gflags::GetArgv0(), true);
  }

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

  /* ===================== Create Dataset ===================== */
  if (FLAGS_emission_dir.empty()) {
    // Load dataset
    int worldRank = 0;
    int worldSize = 1;
    auto ds =
        createDataset(FLAGS_test, dicts, lexicon, 1, worldRank, worldSize);

    ds->shuffle(3);
    LOG(INFO) << "[Serialization] Running forward pass ...";

    int cnt = 0;
    for (auto& sample : *ds) {
      auto rawEmission =
          network->forward({fl::input(sample[kInputIdx])}).front();
      int N = rawEmission.dims(0);
      int T = rawEmission.dims(1);

      auto emission = afToVector<float>(rawEmission);
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

      emissionSet.emissions.emplace_back(emission);
      emissionSet.wordTargets.emplace_back(wordTargetStr);
      emissionSet.tokenTargets.emplace_back(tokenTarget);
      emissionSet.emissionT.emplace_back(T);
      emissionSet.emissionN = N;

      // while decoding we use batchsize 1 and hence ds only has 1 sampleid
      emissionSet.sampleIds.emplace_back(
          readSampleIds(sample[kSampleIdx]).front());

      ++cnt;
      if (cnt == FLAGS_maxload) {
        break;
      }
    }
    if (FLAGS_criterion == kAsgCriterion) {
      emissionSet.transition = afToVector<float>(criterion->param(0).array());
    }
  }

  int nSample = emissionSet.emissions.size();
  nSample = FLAGS_maxload > 0 ? std::min(nSample, FLAGS_maxload) : nSample;
  int nSamplePerThread =
      std::ceil(nSample / static_cast<float>(FLAGS_nthread_decoder));
  LOG(INFO) << "[Dataset] Number of samples per thread: " << nSamplePerThread;

  network.reset(); // AM is only used in running forward pass. So we will free
                   // the space of it on GPU or memory. network.use_count() will
                   // be 0 after this call.
  af::deviceGC();
  /* ===================== Decode ===================== */
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
  } else if (FLAGS_criterion == kSeq2SeqCriterion) {
    criterionType = CriterionType::S2S;
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion;
  }

  const auto& transition = emissionSet.transition;

  // Prepare decoder options
  DecoderOptions decoderOpt(
      FLAGS_beamsize,
      FLAGS_beamsizetoken,
      static_cast<float>(FLAGS_beamthreshold),
      static_cast<float>(FLAGS_lmweight),
      static_cast<float>(FLAGS_wordscore),
      static_cast<float>(FLAGS_unkscore),
      static_cast<float>(FLAGS_silscore),
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

  auto writeHyp = [&](const std::string& hypStr) {
    std::lock_guard<std::mutex> lock(hypMutex);
    hypStream << hypStr;
  };
  auto writeRef = [&](const std::string& refStr) {
    std::lock_guard<std::mutex> lock(refMutex);
    refStream << refStr;
  };
  auto writeLog = [&](const std::string& logStr) {
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
  int silIdx = tokenDict.getIndex(FLAGS_wordseparator);
  std::shared_ptr<Trie> trie = nullptr;
  if (FLAGS_uselexicon) {
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

  // Decoding
  auto runDecoder = [&](int tid, int start, int end) {
    try {
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

      // Build Decoder
      std::unique_ptr<Decoder> decoder;
      if (FLAGS_decodertype == "wrd") {
        decoder.reset(new WordLMDecoder(
            decoderOpt,
            trie,
            localLm,
            silIdx,
            blankIdx,
            unkWordIdx,
            transition));
        LOG(INFO) << "[Decoder] Decoder with word-LM loaded in thread: " << tid;
      } else if (FLAGS_decodertype == "tkn") {
        if (criterionType == CriterionType::S2S) {
          auto amUpdateFunc = buildAmUpdateFunction(localCriterion);
          int eosIdx = tokenDict.getIndex(kEosToken);

          decoder.reset(new Seq2SeqDecoder(
              decoderOpt,
              localLm,
              eosIdx,
              amUpdateFunc,
              FLAGS_maxdecoderoutputlen,
              static_cast<float>(FLAGS_hardselection),
              static_cast<float>(FLAGS_softselection)));
          LOG(INFO)
              << "[Decoder] Seq2Seq decoder with token-LM loaded in thread: "
              << tid;
        } else if (FLAGS_uselexicon) {
          decoder.reset(new TokenLMDecoder(
              decoderOpt,
              trie,
              localLm,
              silIdx,
              blankIdx,
              unkWordIdx,
              transition));
          LOG(INFO) << "[Decoder] Decoder with token-LM loaded in thread: "
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

      // Get data and run decoder
      TestMeters meters;
      int sliceSize = end - start;
      meters.timer.resume();
      for (int s = start; s < end; s++) {
        auto emission = emissionSet.emissions[s];
        auto wordTarget = emissionSet.wordTargets[s];
        auto tokenTarget = emissionSet.tokenTargets[s];
        auto sampleId = emissionSet.sampleIds[s];
        auto T = emissionSet.emissionT[s];
        auto N = emissionSet.emissionN;

        // DecodeResult
        auto results = decoder->decode(emission.data(), T, N);

        // Cleanup predictions
        auto& rawWordPrediction = results[0].words;
        auto& rawTokenPrediction = results[0].tokens;

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

        // Update meters & print out predictions
        meters.werSlice.add(wordPrediction, wordTarget);
        meters.lerSlice.add(letterPrediction, letterTarget);

        auto wordTargetStr = join(" ", wordTarget);
        auto wordPredictionStr = join(" ", wordPrediction);
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
                 << "\%, progress (slice " << tid
                 << "): " << static_cast<float>(s - start + 1) / sliceSize * 100
                 << "\%]" << std::endl;

          std::cout << buffer.str();
          if (!FLAGS_sclite.empty()) {
            writeLog(buffer.str());
          }
        }

        // Update conters
        sliceNumWords[tid] += wordTarget.size();
        sliceNumTokens[tid] += letterTarget.size();
      }
      meters.timer.stop();
      sliceWer[tid] = meters.werSlice.value()[0];
      sliceLer[tid] = meters.lerSlice.value()[0];
      sliceNumSamples[tid] = sliceSize;
      sliceTime[tid] = meters.timer.value();
    } catch (const std::exception& exc) {
      LOG(FATAL) << "Exception in thread " << tid << "\n" << exc.what();
    }
  };

  /* Spread threades */
  auto startThreads = [&]() {
    if (FLAGS_nthread_decoder == 1) {
      runDecoder(0, 0, nSample);
    } else if (FLAGS_nthread_decoder > 1) {
      fl::ThreadPool threadPool(FLAGS_nthread_decoder);
      for (int i = 0; i < FLAGS_nthread_decoder; i++) {
        int start = i * nSamplePerThread;
        if (start >= nSample) {
          break;
        }
        int end = std::min((i + 1) * nSamplePerThread, nSample);
        threadPool.enqueue(runDecoder, i, start, end);
      }
    } else {
      LOG(FATAL) << "Invalid nthread_decoder";
    }
  };
  auto timer = fl::TimeMeter();
  timer.resume();
  startThreads();
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
