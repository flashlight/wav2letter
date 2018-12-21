/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <utility>
#include <vector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Utils.h"
#include "data/NumberedFilesLoader.h"
#include "data/Utils.h"
#include "fb/W2lEverstoreDataset.h"
#include "feature/Sound.h"

using namespace w2l;

std::pair<int, int> getPaddingStats(
    int64_t batchsize,
    const std::vector<int64_t>& sortedSampleIndices,
    const std::vector<SpeechSampleMetaInfo>& samples) {
  double input_pad = 0;
  int output_pad = 0;
  for (int i = 0; i < sortedSampleIndices.size() - batchsize + 1;
       i += batchsize) {
    auto& sample = samples[sortedSampleIndices[i]];

    double input_min, input_max;
    input_min = input_max = sample.audiolength();

    int64_t output_min, output_max;
    output_min = output_max = sample.reflength();

    for (int j = 1; j < batchsize; j++) {
      auto& s = samples[sortedSampleIndices[i + j]];
      output_max = std::max(s.reflength(), output_max);
      output_min = std::min(s.reflength(), output_min);
      input_max = std::max(s.audiolength(), input_max);
      input_min = std::min(s.audiolength(), input_min);
    }
    auto input_diff_frames = (input_max - input_min) / kFrameStrideMs;
    input_pad += input_diff_frames * batchsize;
    output_pad += (output_max - output_min) * batchsize;
  }

  return std::make_pair((int)input_pad, output_pad);
}

void logPaddingStats(
    const std::string& sortfn,
    const std::vector<SpeechSampleMetaInfo>& samples,
    int batchsize,
    int input_bin_size,
    int output_bin_size) {
  auto sortedindices =
      sortSamples(samples, sortfn, input_bin_size, output_bin_size);
  auto paddingStats = getPaddingStats(batchsize, sortedindices, samples);

  LOG(INFO) << "Sort Function: " << sortfn
            << ", Audio Padding: " << std::get<0>(paddingStats)
            << ", Transcript Padding: " << std::get<1>(paddingStats)
            << ", Input Bin: " << input_bin_size
            << ", Output Bin: " << output_bin_size;
}

int findBestParams(std::function<int(int)> sorter) {
  auto best = sorter(1);
  auto best_bin_size = 1;
  for (int i = 2; i < 200; i += 1) {
    auto result = sorter(i);
    if (result < best) {
      best = result;
      best_bin_size = i;
    }
  }
  return best_bin_size;
}

std::vector<SpeechSampleMetaInfo> loadEverstore(const std::string& path) {
  W2lEverstoreDataset::init(); // Required for everstore client

  auto dict = makeDictionary();
  LOG(INFO) << "Number of classes (network) = " << dict.indexSize();

  DictionaryMap dicts;
  dicts.insert({kTargetIdx, dict});
  W2lEverstoreDataset ds(path, dicts, FLAGS_batchsize, 0, 1, FLAGS_targettype);

  std::vector<SpeechSampleMetaInfo> samples;
  for (size_t i = 0; i < ds.getSamples().size(); ++i) {
    const auto& sample = ds.getSamples()[i];
    samples.emplace_back(sample.durationMs, sample.ref.length(), i);
  }
  return samples;
}

std::vector<SpeechSampleMetaInfo> loadLocal(const std::string& path) {
  TargetExtMap targetexts;
  targetexts.insert({kTargetIdx, FLAGS_target});
  NumberedFilesLoader loader(path, FLAGS_input, targetexts);
  std::vector<SpeechSampleMetaInfo> samples;
  for (int64_t i = 0; i < loader.size(); ++i) {
    auto audiofile = loader.filename(i, FLAGS_input);
    auto targetfile = loader.filename(i, FLAGS_target);
    auto info = speech::loadSoundInfo(audiofile.c_str());
    auto durationMs = ((double)info.frames / info.samplerate) * 1e3;
    auto ref = loadTarget(targetfile);
    samples.emplace_back(durationMs, ref.size(), i);
  }
  return samples;
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  if (argc < 2) {
    LOG(FATAL) << "Specify a data path.";
  }

  std::string path = argv[1];
  gflags::ParseCommandLineFlags(&argc, &argv, false);
  int batchsize = FLAGS_batchsize;

  auto samples = FLAGS_everstoredb ? loadEverstore(path) : loadLocal(path);

  logPaddingStats("input", samples, batchsize, 0, 0);

  auto input_spiral_sorter = [&](int output_bin_size) {
    auto stats = getPaddingStats(
        batchsize,
        sortSamples(samples, "input_spiral", 0, output_bin_size),
        samples);
    return std::get<0>(stats) + std::get<1>(stats);
  };

  int best_out_size = findBestParams(input_spiral_sorter);
  logPaddingStats("input_spiral", samples, batchsize, 0, best_out_size);

  auto output_spiral_sorter = [&](int input_bin_size) {
    auto stats = getPaddingStats(
        batchsize,
        sortSamples(samples, "output_spiral", input_bin_size, 0),
        samples);
    return std::get<0>(stats) + std::get<1>(stats);
  };

  int best_in_size = findBestParams(output_spiral_sorter);
  logPaddingStats("output_spiral", samples, batchsize, best_in_size, 0);

  return 0;
}
