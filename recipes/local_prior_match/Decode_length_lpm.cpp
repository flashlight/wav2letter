/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/flashlight.h>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <fstream>
#include <iostream>

#include "common/Defines.h"
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
#include "criterion/criterion.h"
#include "module/module.h"
#include "runtime/runtime.h"

using namespace w2l;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec + " [model] [dataset] [outputfile]");

  if (argc <= 3) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::string reloadpath = argv[1];
  std::string dataset = argv[2];
  std::string outputfile = argv[3];
  std::unordered_map<std::string, std::string> cfg;
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<SequenceCriterion> criterion;

  W2lSerializer::load(reloadpath, cfg, network, criterion);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "Invalid config loaded from " << reloadpath;
  }
  LOG(INFO) << "Reading flags from config file " << reloadpath;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  Dictionary dict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
  if (FLAGS_eostoken) {
    dict.addEntry(kEosToken);
  }

  LOG(INFO) << "Number of classes (network) = " << dict.indexSize();

  LOG(INFO) << "[network] " << network->prettyString();

  af::setSeed(FLAGS_seed);

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});
  auto lexicon = loadWords(FLAGS_lexicon, FLAGS_maxword);

  auto testset = createDataset(dataset, dicts, lexicon, 1, 0, 1);

  network->eval();
  criterion->eval();

  std::ofstream out;
  out.open(outputfile);

  for (auto& sample : *testset) {
    auto sampleId = readSampleIds(sample[kSampleIdx]).front();
    auto output = network->forward({fl::input(sample[kInputIdx])}).front();

    auto viterbipathArr = criterion->viterbiPath(output.array());
    auto viterbipath = afToVector<int>(viterbipathArr);
    remapLabels(viterbipath, dict);

    if (viterbipath.size() == 0) {
      continue;
    }
    // assume "reflen1" is not a valid word in the lexicon
    out << sampleId << " reflen" << std::to_string(viterbipath.size())
        << std::endl;
  }
  out.close();

  return 0;
}
