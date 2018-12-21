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
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lDataset.h"
#include "data/W2lNumberedFilesDataset.h"
#include "module/module.h"
#include "runtime/Logger.h"
#include "runtime/Serial.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
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
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  /* ===================== Create Network ===================== */
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::SequenceCriterion> criterion;
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
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto word2spell = loadWords(FLAGS_lexicon, FLAGS_maxword);

  auto letterDict = makeDictionary(FLAGS_tokens);
  int numClasses = letterDict.indexSize();
  LOG(INFO) << "Number of classes (network): " << numClasses;

  Dictionary wordDict;
  for (auto& it : word2spell) {
    wordDict.addToken(it.first);
  }
  wordDict.setDefaultIndex(wordDict.getIndex(kUnkToken));
  LOG(INFO) << "Number of words: " << wordDict.indexSize();

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, letterDict});
  dicts.insert({kWordIdx, wordDict});

  /* ===================== Create Dataset ===================== */
  int worldRank = 0;
  int worldSize = 1;
  W2lNumberedFilesDataset ds(
      FLAGS_test, dicts, 1, worldRank, worldSize, FLAGS_datadir);
  ds.shuffle(3);
  int nSamples = ds.size();
  if (FLAGS_maxload > 0) {
    nSamples = std::min(nSamples, FLAGS_maxload);
  }
  LOG(INFO) << "[Dataset] Dataset loaded.";

  /* ===================== Test ===================== */
  TestMeters meters;

  auto emissionSet = std::make_shared<EmissionSet>();
  meters.timer.resume();
  int cnt = 1;
  for (auto& sample : ds) {
    auto rawEmission = network->forward(fl::input(sample[kInputIdx]));
    auto emission = afToVector<float>(rawEmission);
    auto ltrTarget = afToVector<int>(sample[kTargetIdx]);
    auto wrdTarget = afToVector<int>(sample[kWordIdx]);

    /* viterbiPath + remove duplication/blank */
    auto viterbiPath =
        afToVector<int>(criterion->viterbiPath(rawEmission.array()));
    if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kAsgCriterion) {
      uniq(viterbiPath);
    }
    if (FLAGS_criterion == kCtcCriterion) {
      auto blankidx = letterDict.getIndex(kBlankToken);
      viterbiPath.erase(
          std::remove(viterbiPath.begin(), viterbiPath.end(), blankidx),
          viterbiPath.end());
    }
    remapLabels(viterbiPath, letterDict);
    remapLabels(ltrTarget, letterDict);

    meters.lerSlice.add(viterbiPath, ltrTarget);

    auto wordViterbi = ltrTensor2wrdTensor(
        viterbiPath, wordDict, letterDict, letterDict.getIndex(kSilToken));

    meters.werSlice.add(wordViterbi, wrdTarget);

    if (FLAGS_show) {
      meters.ler.reset();
      meters.wer.reset();
      meters.ler.add(viterbiPath, ltrTarget);
      meters.wer.add(wordViterbi, wrdTarget);

      std::cout << "|T|: " << tensor2letters(ltrTarget, letterDict)
                << std::endl;
      std::cout << "|P|: " << tensor2letters(viterbiPath, letterDict)
                << std::endl;
      std::cout << "[sample: " << cnt << ", WER: " << meters.wer.value()[0]
                << "\%, LER: " << meters.ler.value()[0]
                << "\%, total WER: " << meters.werSlice.value()[0]
                << "\%, total LER: " << meters.lerSlice.value()[0]
                << "\%, progress: " << static_cast<float>(cnt) / nSamples * 100
                << "\%]" << std::endl;
      ++cnt;
      if (cnt == FLAGS_maxload) {
        break;
      }
    }

    /* Save emission and targets */
    int N = rawEmission.dims(0);
    int T = rawEmission.dims(1);
    emissionSet->emissions.emplace_back(emission);
    emissionSet->letterTargets.emplace_back(ltrTarget);
    emissionSet->wordTargets.emplace_back(wrdTarget);
    emissionSet->emissionT.emplace_back(T);
    emissionSet->emissionN = N;
  }
  if (FLAGS_criterion == kAsgCriterion) {
    emissionSet->transition = afToVector<float>(criterion->param(0).array());
  }
  meters.timer.stop();
  std::cout << "---\n[total WER: " << meters.werSlice.value()[0]
            << "\%, total LER: " << meters.lerSlice.value()[0]
            << "\%, time: " << meters.timer.value() << "s]" << std::endl;

  /* ====== Serialize emission and targets for decoding ====== */
  std::string savePath = pathsConcat(FLAGS_emission_dir, FLAGS_test + ".bin");
  LOG(INFO) << "[Serialization] Saving into file: " << savePath;
  W2lSerializer::save(savePath, emissionSet);

  return 0;
}
