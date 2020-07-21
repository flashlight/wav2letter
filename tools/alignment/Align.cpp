/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "criterion/criterion.h"
#include "libraries/common/Dictionary.h"
#include "module/module.h"
#include "runtime/runtime.h"

#include "tools/alignment/Utils.h"

using namespace w2l;
using namespace w2l::alignment;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);
  std::vector<std::string> argvs;
  for (int i = 0; i < argc; i++) {
    argvs.emplace_back(argv[i]);
  }
  gflags::SetUsageMessage("Usage: \n " + exec + " align_file_path [flags]");
  if (argc <= 2) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::string alignFilePath = argv[1];

  /* ===================== Parse Options ===================== */
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
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
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  if (!FLAGS_flagsfile.empty()) {
    LOG(INFO) << "Reading flags from file " << FLAGS_flagsfile;
    gflags::ReadFromFlagsFile(FLAGS_flagsfile, argv[0], true);
  }

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  /* ===================== Create Dictionary ===================== */
  auto dictPath = pathsConcat(FLAGS_tokensdir, FLAGS_tokens);
  LOG(INFO) << "Loading dictionary from " << dictPath;
  if (dictPath.empty() || !fileExists(dictPath)) {
    throw std::invalid_argument("Invalid dictionary filepath specified.");
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

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, tokenDict});

  std::mutex write_mutex;
  std::ofstream alignFile;
  alignFile.open(alignFilePath);
  if (!alignFile.is_open() || !alignFile.good()) {
    LOG(FATAL) << "Error opening log file";
  } else {
    LOG(INFO) << "Writing alignment to: " << alignFilePath;
  }

  LexiconMap lexicon;
  if (!FLAGS_lexicon.empty()) {
    lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);
  }

  LOG(INFO) << "Loaded lexicon";

  auto writeLog = [&](const std::string& logStr) {
    std::lock_guard<std::mutex> lock(write_mutex);
    alignFile << logStr;
    if (FLAGS_show) {
      std::cout << logStr;
    }
  };

  /* ===================== Create Dataset ===================== */
  int worldRank = 0;
  int worldSize = 1;
  std::shared_ptr<W2lDataset> ds;
  ds = createDataset(
      FLAGS_test, dicts, lexicon, FLAGS_batchsize, worldRank, worldSize);

  LOG(INFO) << "[Dataset] Dataset loaded";

  auto postprocessFN = getWordSegmenter(criterion);

  int batches = 0;
  fl::TimeMeter alignMtr;
  fl::TimeMeter fwdMtr;
  fl::TimeMeter parseMtr;

  for (auto& sample : *ds) {
    fwdMtr.resume();
    const auto input = fl::input(sample[kInputIdx]);
    std::vector<fl::Variable> rawEmissions = network->forward({input});
    fl::Variable rawEmission;
    if (!rawEmissions.empty()) {
      rawEmission = rawEmissions.front();
    } else {
      LOG(ERROR) << "Network did not produce any outputs";
    }

    fwdMtr.stop();
    alignMtr.resume();
    auto bestPaths =
        criterion->viterbiPath(rawEmission.array(), sample[kTargetIdx]);
    alignMtr.stop();
    parseMtr.resume();

    const double timeScale =
        static_cast<double>(input.dims(0)) / rawEmission.dims(1);

    const std::vector<std::vector<std::string>> tokenPaths =
        mapIndexToToken(bestPaths, dicts);
    const std::vector<std::string> sampleIdsStr =
        readSampleIds(sample[kSampleIdx]);

    for (int b = 0; b < tokenPaths.size(); b++) {
      if (sampleIdsStr.size() > b) {
        const std::vector<std::string>& path = tokenPaths[b];
        const std::vector<AlignedWord> segmentation = postprocessFN(
            path, FLAGS_replabel, FLAGS_framestridems * timeScale);
        const std::string ctmString = getCTMFormat(segmentation);
        std::stringstream buffer;
        buffer << sampleIdsStr[b] << "\t" << ctmString << "\n";
        writeLog(buffer.str());
      }
    }
    parseMtr.stop();
    ++batches;
    if (batches % 500 == 0) {
      LOG(INFO) << "Done batches: " << batches
                << " , samples: " << batches * FLAGS_batchsize;
    }
  }

  LOG(INFO) << "Align time: " << alignMtr.value();
  LOG(INFO) << "Fwd time: " << fwdMtr.value();
  LOG(INFO) << "Parse time: " << parseMtr.value();
  alignFile.close();
  return 0;
}
