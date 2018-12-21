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

#include "common/Defines.h"
#include "common/Transforms.h"
#include "common/Utils.h"
#include "criterion/criterion.h"
#include "data/W2lNumberedFilesDataset.h"
#include "module/module.h"
#include "runtime/Serial.h"

using namespace w2l;

namespace w2l {

DEFINE_bool(
    viewtranscripts,
    false,
    "Log the Reference and Hypothesis transcripts.");

DEFINE_string(attndir, "", "Directory for attention output.");

DEFINE_int64(beamsz, 1, "Size of beam for beam search.");

} // namespace w2l

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  std::string exec(argv[0]);

  gflags::SetUsageMessage(
      "Usage: \n " + exec +
      " [model] [dataset], optional: --attndir=[directory]");

  if (argc <= 2) {
    LOG(FATAL) << gflags::ProgramUsage();
  }

  std::string reloadpath = argv[1];
  std::string dataset = argv[2];
  std::unordered_map<std::string, std::string> cfg;
  std::shared_ptr<fl::Module> network;
  std::shared_ptr<fl::SequenceCriterion> base_criterion;

  W2lSerializer::load(reloadpath, cfg, network, base_criterion);
  auto criterion =
      std::dynamic_pointer_cast<fl::Seq2SeqCriterion>(base_criterion);

  LOG(INFO) << "Parsing command line flags";
  gflags::ParseCommandLineFlags(&argc, &argv, false);

  auto flags = cfg.find(kGflags);
  if (flags == cfg.end()) {
    LOG(FATAL) << "Invalid config loaded from " << reloadpath;
  }
  LOG(INFO) << "Reading flags from config file " << reloadpath;
  gflags::ReadFlagsFromString(flags->second, gflags::GetArgv0(), true);
  LOG(INFO) << "Gflags after parsing \n" << serializeGflags("; ");

  Dictionary dict = makeDictionary();

  LOG(INFO) << "Number of classes (network) = " << dict.indexSize();

  LOG(INFO) << "[network] " << network->prettyString();

  af::setMemStepSize(FLAGS_memstepsize);
  af::setSeed(FLAGS_seed);
  std::string metername = FLAGS_target == "ltr" ? "LER: " : "PER: ";

  fl::EditDistanceMeter cerMeter;
  fl::EditDistanceMeter cerMeter_single;
  fl::EditDistanceMeter cerBeamMeter;
  fl::EditDistanceMeter werBeamMeter;
  fl::EditDistanceMeter cerTfMeter;
  fl::AverageValueMeter lossMeter;

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});

  W2lNumberedFilesDataset testset(
      dataset, dicts, FLAGS_batchsize, 0, 1, FLAGS_datadir);

  auto toString = [&dict](const std::vector<int>& transcript) {
    std::stringstream ss;
    for (auto l : transcript) {
      ss << dict.getToken(l);
      if (FLAGS_target == "phn") {
        ss << " ";
      }
    }
    return ss.str();
  };

  network->eval();
  criterion->eval();

  int uid = 1;
  for (auto& sample : testset) {
    auto output = network->forward(fl::input(sample[kInputIdx]));
    auto target = sample[kTargetIdx];
    auto tfout =
        std::get<0>(criterion->vectorizedDecoder(output, fl::noGrad(target)));
    af::array maxv, argmaxid;
    af::max(maxv, argmaxid, tfout.array(), 0);
    argmaxid = argmaxid.as(s32);

    auto loss = criterion->forward(output, fl::noGrad(target));
    auto lossvec = afToVector<float>(loss.array());
    for (int b = 0; b < output.dims(2); ++b) {
      auto tgt = target(af::span, b);
      auto teacherpath = afToVector<int>(argmaxid(af::span, af::span, b));
      auto tgtraw = afToVector<int>(tgt);

      fl::Variable attention;
      af::array viterbipathArr;
      std::tie(viterbipathArr, attention) = criterion->viterbiPathBase(
          output.array()(af::span, af::span, b), (FLAGS_attndir != ""));

      auto viterbipath = afToVector<int>(viterbipathArr);
      auto beampath = viterbipath;
      if (FLAGS_beamsz > 1) {
        beampath = criterion->beamPath(
            output.array()(af::span, af::span, b), FLAGS_beamsz);
      }

      // Truncate EOS tokens in teacherpath
      auto eosloc = std::find(
          teacherpath.begin(), teacherpath.end(), dict.getIndex(kEosToken));
      teacherpath.resize(eosloc - teacherpath.begin());

      remapLabels(beampath, dict);
      remapLabels(viterbipath, dict);
      remapLabels(tgtraw, dict);

      cerMeter.add(viterbipath, tgtraw);
      cerBeamMeter.add(beampath, tgtraw);
      if (FLAGS_target == "ltr") {
        auto beamwords = split("|", toString(beampath), true);
        auto tgtwords = split("|", toString(tgtraw), true);
        werBeamMeter.add(beamwords, tgtwords);
      }
      cerTfMeter.add(teacherpath, tgtraw);
      lossMeter.add(lossvec[b]);

      if (FLAGS_viewtranscripts) {
        cerMeter_single.reset();
        cerMeter_single.add(viterbipath, tgtraw);

        std::cout << "UID: " << uid << ", " << metername
                  << cerMeter_single.value()[0]
                  << ", DEL: " << cerMeter_single.value()[2]
                  << ", INS: " << cerMeter_single.value()[3]
                  << ", SUB: " << cerMeter_single.value()[4] << std::endl;
        std::cout << "REF     ";
        std::cout << toString(tgtraw) << std::endl;

        std::cout << "BEAM HYP  ";
        std::cout << toString(beampath) << std::endl;

        std::cout << "VP HYP  ";
        std::cout << toString(viterbipath) << std::endl;

        std::cout << "TF HYP  ";
        std::cout << toString(teacherpath) << std::endl;

        std::cout << "===============" << std::endl;
      }

      if (FLAGS_attndir != "") {
        auto filename = FLAGS_attndir + "/" + std::to_string(uid) + "_attn.out";
        std::string key(std::to_string(uid));
        std::for_each(
            viterbipath.begin(), viterbipath.end(), [&](const int& s) {
              key += ("-" + dict.getToken(s));
            });
        key += "-<eos>";
        af::saveArray(key.c_str(), attention.array(), filename.c_str(), false);
      }
      ++uid;
    }
  }

  auto beamcer = cerBeamMeter.value()[0];
  auto beamdel = cerBeamMeter.value()[2];
  auto beamins = cerBeamMeter.value()[3];
  auto beamsub = cerBeamMeter.value()[4];
  auto viterbicer = cerMeter.value()[0];
  auto viterbidel = cerMeter.value()[2];
  auto viterbiins = cerMeter.value()[3];
  auto viterbisub = cerMeter.value()[4];
  auto tfcer = cerTfMeter.value()[0];
  auto avgloss = lossMeter.value()[0];
  LOG(INFO) << "Beam Search " << metername << beamcer << ", DEL: " << beamdel
            << ", INS: " << beamins << ", SUB: " << beamsub;
  LOG(INFO) << "Viterbi " << metername << viterbicer << ", DEL: " << viterbidel
            << ", INS: " << viterbiins << ", SUB: " << viterbisub;
  LOG(INFO) << "Teacher Forced " << metername << tfcer << ", Loss: " << avgloss;

  if (FLAGS_target == "ltr") {
    auto beamwer = werBeamMeter.value()[0];
    LOG(INFO) << "Beam Search WER " << beamwer;
  }

  return 0;
}
