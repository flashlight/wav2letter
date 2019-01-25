/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
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
#include "common/Dictionary.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/Featurize.h"
#include "data/W2lDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "decoder/Decoder.hpp"
#include "decoder/KenLM.hpp"
#include "decoder/Trie.hpp"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"

#ifdef BUILD_FB_DEPENDENCIES
#include "fb/W2lEverstoreDataset.h"
#endif

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
      "Usage: \n " + exec + " [data_path] [dataset_name] [flags]");
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
  }

  /* ===================== Create Network ===================== */
  if (!(FLAGS_am.empty() ^ FLAGS_emission_dir.empty())) {
    LOG(FATAL)
        << "One and only one of flag -am and -emission_dir should be set.";
  }
  EmissionSet emissionSet;

  /* Using acoustic model */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  if (!FLAGS_am.empty()) {
    std::unordered_map<std::string, std::string> cfg;
    LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;

    W2lSerializer::load(FLAGS_am, cfg, network, criterion);
    network->eval();
    LOG(INFO) << "[Network] " << network->prettyString();
    if (criterion) {
      criterion->eval();
      LOG(INFO) << "[Network] " << criterion->prettyString();
    }
    LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

    auto flags = cfg.find(kGflags);
    if (flags == cfg.end()) {
      LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
    }
    LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
    gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
  }
  /* Using existing emissions */
  else {
    std::string cleanedTestPath = cleanFilepath(FLAGS_test);
    std::string loadPath =
        pathsConcat(FLAGS_emission_dir, cleanedTestPath + ".bin");
    LOG(INFO) << "[Serialization] Loading file: " << loadPath;
    W2lSerializer::load(loadPath, emissionSet);
    gflags::ReadFlagsFromString(emissionSet.gflags, gflags::GetArgv0(), true);
  }

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */

  auto tokenDict = createTokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  int numClasses = tokenDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  auto wordDict = createWordDict(lexicon);
  LOG(INFO) << "Number of words: " << wordDict.indexSize();

  DictionaryMap dicts = {{kTargetIdx, tokenDict}, {kWordIdx, wordDict}};

  /* ===================== Create Dataset ===================== */
  if (!FLAGS_am.empty()) {
    // Load dataset
    int worldRank = 0;
    int worldSize = 1;
    std::unique_ptr<W2lDataset> ds;
    if (FLAGS_everstoredb) {
#ifdef BUILD_FB_DEPENDENCIES
      W2lEverstoreDataset::init(); // Required for everstore client
      ds = std::unique_ptr<W2lEverstoreDataset>(new W2lEverstoreDataset(
          FLAGS_test, dicts, 1, worldRank, worldSize, FLAGS_targettype));
#else
      LOG(FATAL) << "W2lEverstoreDataset not supported: "
                 << "build with -DBUILD_FB_DEPENDENCIES";
#endif
    } else {
      ds = std::unique_ptr<W2lNumberedFilesDataset>(new W2lNumberedFilesDataset(
          FLAGS_test, dicts, 1, worldRank, worldSize, FLAGS_datadir));
    }
    ds->shuffle(3);
    LOG(INFO) << "[Serialization] Running forward pass ...";

    int cnt = 0;
    for (auto& sample : *ds) {
      auto rawEmission =
          network->forward({fl::input(sample[kInputIdx])}).front();
      int N = rawEmission.dims(0);
      int T = rawEmission.dims(1);

      auto emission = afToVector<float>(rawEmission);
      auto ltrTarget = afToVector<int>(sample[kTargetIdx]);
      auto wrdTarget = afToVector<int>(sample[kWordIdx]);
      emissionSet.emissions.emplace_back(emission);
      emissionSet.wordTargets.emplace_back(wrdTarget);
      emissionSet.letterTargets.emplace_back(ltrTarget);
      emissionSet.emissionT.emplace_back(T);
      emissionSet.emissionN = N;
      if (FLAGS_criterion == kAsgCriterion) {
        emissionSet.transition = afToVector<float>(criterion->param(0).array());
      }

      // while decoding we use batchsize 1 and hence ds only has 1 sampleid
      emissionSet.sampleIds.emplace_back(
          afToVector<std::string>(sample[kFileIdIdx]).front());

      ++cnt;
      if (cnt == FLAGS_maxload) {
        break;
      }
    }
  }

  int nSample = emissionSet.emissions.size();
  nSample = FLAGS_maxload > 0 ? std::min(nSample, FLAGS_maxload) : nSample;
  int nSamplePerThread =
      std::ceil(nSample / static_cast<float>(FLAGS_nthread_decoder));
  LOG(INFO) << "[Dataset] Number of samples per thread: " << nSamplePerThread;

  /* ===================== Decode ===================== */
  // Prepare counters
  std::vector<double> sliceWer(FLAGS_nthread_decoder);
  std::vector<double> sliceLer(FLAGS_nthread_decoder);
  std::vector<int> sliceNumWords(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumLetters(FLAGS_nthread_decoder, 0);
  std::vector<int> sliceNumSamples(FLAGS_nthread_decoder, 0);
  std::vector<double> sliceTime(FLAGS_nthread_decoder, 0);

  // Prepare criterion
  ModelType modelType = ModelType::ASG;
  if (FLAGS_criterion == kCtcCriterion) {
    modelType = ModelType::CTC;
  } else if (FLAGS_criterion != kAsgCriterion) {
    LOG(FATAL) << "[Decoder] Invalid model type: " << FLAGS_criterion;
  }

  const auto& transition = emissionSet.transition;

  // Prepare decoder options
  DecoderOptions decoderOpt(
      FLAGS_beamsize,
      static_cast<float>(FLAGS_beamscore),
      static_cast<float>(FLAGS_lmweight),
      static_cast<float>(FLAGS_wordscore),
      static_cast<float>(FLAGS_unkweight),
      FLAGS_forceendsil,
      FLAGS_logadd,
      static_cast<float>(FLAGS_silweight),
      modelType);

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

  // Decoding
  auto runDecoder = [&](int tid, int start, int end) {
    try {
      // Build Language Model
      std::shared_ptr<LM> lm;
      if (FLAGS_lmtype == "kenlm") {
        lm = std::make_shared<KenLM>(FLAGS_lm);
        if (!lm) {
          LOG(FATAL) << "[LM constructing] Failed to load LM: " << FLAGS_lm;
        }
      } else {
        LOG(FATAL) << "[LM constructing] Invalid LM Type: " << FLAGS_lmtype;
      }
      LOG(INFO) << "[Decoder] LM constructed.\n";

      // Build Trie
      if (std::strlen(kSilToken) != 1) {
        LOG(FATAL) << "[Decoder] Invalid unknown_symbol: " << kSilToken;
      }
      if (std::strlen(kBlankToken) != 1) {
        LOG(FATAL) << "[Decoder] Invalid unknown_symbol: " << kBlankToken;
      }
      int silIdx = tokenDict.getIndex(kSilToken);
      int blankIdx = FLAGS_criterion == kCtcCriterion
          ? tokenDict.getIndex(kBlankToken)
          : -1;
      int unkIdx = lm->index(kUnkToken);
      std::shared_ptr<Trie> trie =
          std::make_shared<Trie>(tokenDict.indexSize(), silIdx);
      auto start_state = lm->start(false);

      for (auto& it : lexicon) {
        std::string word = it.first;
        int lmIdx = lm->index(word);
        if (lmIdx == unkIdx) { // We don't insert unknown words
          continue;
        }
        float score;
        auto dummyState = lm->score(start_state, lmIdx, score);
        for (auto& tokens : it.second) {
          auto tokensTensor = tokens2Tensor(tokens, tokenDict);
          trie->insert(
              tokensTensor,
              std::make_shared<TrieLabel>(lmIdx, wordDict.getIndex(word)),
              score);
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

      // Build Decoder
      std::shared_ptr<TrieLabel> unk =
          std::make_shared<TrieLabel>(unkIdx, wordDict.getIndex(kUnkToken));
      Decoder decoder(trie, lm, silIdx, blankIdx, unk);
      LOG(INFO) << "[Decoder] Decoder loaded in thread: " << tid;

      // Get data and run decoder
      TestMeters meters;
      int sliceSize = end - start;
      meters.timer.resume();
      for (int s = start; s < end; s++) {
        auto emission = emissionSet.emissions[s];
        auto wordTarget = emissionSet.wordTargets[s];
        auto letterTarget = emissionSet.letterTargets[s];
        auto sampleId = emissionSet.sampleIds[s];
        auto T = emissionSet.emissionT[s];
        auto N = emissionSet.emissionN;

        std::vector<float> score;
        std::vector<std::vector<int>> wordPredictions;
        std::vector<std::vector<int>> letterPredictions;

        std::tie(score, wordPredictions, letterPredictions) = decoder.decode(
            decoderOpt, transition.data(), emission.data(), T, N);

        // Cleanup predictions
        auto wordPrediction = wordPredictions[0];
        auto letterPrediction = letterPredictions[0];
        if (FLAGS_criterion == kCtcCriterion ||
            FLAGS_criterion == kAsgCriterion) {
          uniq(letterPrediction);
        }
        if (FLAGS_criterion == kCtcCriterion) {
          letterPrediction.erase(
              std::remove(
                  letterPrediction.begin(), letterPrediction.end(), blankIdx),
              letterPrediction.end());
        }
        remapLabels(letterTarget, tokenDict);
        remapLabels(letterPrediction, tokenDict);
        validateTokens(wordPrediction, wordDict.getIndex(kUnkToken));

        // Update meters & print out predictions
        meters.werSlice.add(wordPrediction, wordTarget);
        meters.lerSlice.add(letterPrediction, letterTarget);

        if (FLAGS_show) {
          meters.wer.reset();
          meters.ler.reset();
          meters.wer.add(wordPrediction, wordTarget);
          meters.ler.add(letterPrediction, letterTarget);

          auto wordTargetStr = tensor2words(wordTarget, wordDict);
          auto wordPredictionStr = tensor2words(wordPrediction, wordDict);

          std::stringstream buffer;
          buffer << "|T|: " << wordTargetStr << std::endl;
          buffer << "|P|: " << wordPredictionStr << std::endl;
          if (FLAGS_showletters) {
            buffer << "|t|: " << tensor2letters(letterTarget, tokenDict)
                   << std::endl;
            buffer << "|p|: " << tensor2letters(letterPrediction, tokenDict)
                   << std::endl;
          }
          buffer << "[sample: " << sampleId
                 << ", WER: " << meters.wer.value()[0]
                 << "\%, LER: " << meters.ler.value()[0]
                 << "\%, slice WER: " << meters.werSlice.value()[0]
                 << "\%, slice LER: " << meters.lerSlice.value()[0]
                 << "\%, progress: "
                 << static_cast<float>(s - start + 1) / sliceSize * 100 << "\%]"
                 << std::endl;

          std::cout << buffer.str();
          if (!FLAGS_sclite.empty()) {
            std::string suffix = "(" + sampleId + ")\n";
            writeHyp(wordPredictionStr + suffix);
            writeRef(wordTargetStr + suffix);
            writeLog(buffer.str());
          }
        }

        // Update conters
        sliceNumWords[tid] += wordTarget.size();
        sliceNumLetters[tid] += letterTarget.size();
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
    fl::ThreadPool threadPool(FLAGS_nthread_decoder);
    for (int i = 0; i < FLAGS_nthread_decoder; i++) {
      int start = i * nSamplePerThread;
      if (start >= nSample) {
        break;
      }
      int end = std::min((i + 1) * nSamplePerThread, nSample);
      threadPool.enqueue(runDecoder, i, start, end);
    }
  };
  auto timer = fl::TimeMeter();
  timer.resume();
  startThreads();
  timer.stop();

  /* Compute statistics */
  int totalLetters = 0, totalWords = 0, totalSamples = 0;
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    totalLetters += sliceNumLetters[i];
    totalWords += sliceNumWords[i];
    totalSamples += sliceNumSamples[i];
  }
  double totalWer = 0, totalLer = 0, totalTime = 0;
  for (int i = 0; i < FLAGS_nthread_decoder; i++) {
    totalWer += sliceWer[i] * sliceNumWords[i] / totalWords;
    totalLer += sliceLer[i] * sliceNumLetters[i] / totalLetters;
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
