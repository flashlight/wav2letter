/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Dictionary.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
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
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;
  std::unordered_map<std::string, std::string> cfg;
  LOG(INFO) << "[Network] Reading acoustic model from " << FLAGS_am;
  W2lSerializer::load(FLAGS_am, cfg, network, criterion);
  network->eval();
  criterion->eval();

  LOG(INFO) << "[Network] " << network->prettyString();
  LOG(INFO) << "[Criterion] " << criterion->prettyString();
  LOG(INFO) << "[Network] Number of params: " << numTotalParams(network);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "[Network] Invalid config loaded from " << FLAGS_am;
  }
  LOG(INFO) << "[Network] Updating flags from config file: " << FLAGS_am;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);

  // override with user-specified flags
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!flagsfile.empty()) {
    gflags::ReadFromFlagsFile(flagsfile, argv[0], true);
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
  // Load dataset
  int worldRank = 0;
  int worldSize = 1;
  auto ds = createDataset(FLAGS_test, dicts, lexicon, 1, worldRank, worldSize);

  ds->shuffle(3);
  int nSamples = ds->size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }
  LOG(INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Test ===================== */
  TestMeters meters;

  EmissionSet emissionSet;
  meters.timer.resume();
  int cnt = 0;
  for (auto& sample : *ds) {
    auto rawEmission = network->forward({fl::input(sample[kInputIdx])}).front();
    auto emission = afToVector<float>(rawEmission);
    auto tokenTarget = afToVector<int>(sample[kTargetIdx]);
    auto wordTarget = afToVector<int>(sample[kWordIdx]);
    auto sampleId = readSampleIds(sample[kSampleIdx]).front();

    auto letterTarget = tknTarget2Ltr(tokenTarget, tokenDict);
    std::vector<std::string> wordTargetStr;
    if (!FLAGS_lexicon.empty() && FLAGS_criterion != kSeq2SeqCriterion) {
      wordTargetStr = wrdIdx2Wrd(wordTarget, wordDict);
    } else {
      wordTargetStr = tkn2Wrd(letterTarget);
    }

    // Tokens
    auto tokenPrediction =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
    auto letterPrediction = tknPrediction2Ltr(tokenPrediction, tokenDict);

    meters.lerSlice.add(letterPrediction, letterTarget);

    // Words
    std::vector<std::string> wrdPredictionStr = tkn2Wrd(letterPrediction);
    meters.werSlice.add(wrdPredictionStr, wordTargetStr);

    if (FLAGS_show) {
      meters.ler.reset();
      meters.wer.reset();
      meters.ler.add(letterPrediction, letterTarget);
      meters.wer.add(wrdPredictionStr, wordTargetStr);

      std::cout << "|T|: " << join(" ", letterTarget) << std::endl;
      std::cout << "|P|: " << join(" ", letterPrediction) << std::endl;
      std::cout << "[sample: " << sampleId << ", WER: " << meters.wer.value()[0]
                << "\%, LER: " << meters.ler.value()[0]
                << "\%, total WER: " << meters.werSlice.value()[0]
                << "\%, total LER: " << meters.lerSlice.value()[0]
                << "\%, progress: " << static_cast<float>(cnt) / nSamples * 100
                << "\%]" << std::endl;
    }

    /* Save emission and targets */
    int N = rawEmission.dims(0);
    int T = rawEmission.dims(1);
    emissionSet.emissions.emplace_back(emission);
    emissionSet.tokenTargets.emplace_back(tokenTarget);
    emissionSet.wordTargets.emplace_back(wordTargetStr);

    // while testing we use batchsize 1 and hence ds only has 1 sampleid
    emissionSet.sampleIds.emplace_back(
        readSampleIds(sample[kSampleIdx]).front());

    emissionSet.emissionT.emplace_back(T);
    emissionSet.emissionN = N;

    ++cnt;
    if (cnt == FLAGS_maxload) {
      break;
    }
  }
  if (FLAGS_criterion == kAsgCriterion) {
    emissionSet.transition = afToVector<float>(criterion->param(0).array());
  }
  emissionSet.gflags = serializeGflags();

  meters.timer.stop();
  std::cout << "---\n[total WER: " << meters.werSlice.value()[0]
            << "\%, total LER: " << meters.lerSlice.value()[0]
            << "\%, time: " << meters.timer.value() << "s]" << std::endl;

  /* ====== Serialize emission and targets for decoding ====== */
  std::string cleanedTestPath = cleanFilepath(FLAGS_test);
  std::string savePath =
      pathsConcat(FLAGS_emission_dir, cleanedTestPath + ".bin");
  LOG(INFO) << "[Serialization] Saving into file: " << savePath;
  W2lSerializer::save(savePath, emissionSet);

  return 0;
}
