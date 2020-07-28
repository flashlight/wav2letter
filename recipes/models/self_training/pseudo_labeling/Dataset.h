/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/Utils.h"

namespace filter {
namespace dataset {

struct Sample {
  Sample(
      const std::string& id,
      const std::string& path,
      std::string duration,
      const std::string& transcript,
      const std::vector<std::string>& transcriptWords)
      : id(id),
        path(path),
        duration(duration),
        transcript(transcript),
        transcriptWords(transcriptWords) {}

  std::string toString() const {
    return id + " " + path + " " + duration + " " + transcript;
  }

  size_t getDuration() const {
    return std::stof(duration);
  }

  std::string id;
  std::string path;
  std::string duration;
  std::string transcript;
  std::vector<std::string> transcriptWords;
};

std::unordered_map<std::string, std::shared_ptr<Sample>>
createTranscriptDictFromFile(const std::string& path) {
  std::unordered_map<std::string, std::shared_ptr<Sample>> output;

  if (!w2l::fileExists(path)) {
    throw std::invalid_argument(
        "Input file at path '" + path + "' doesn't exist.");
  }
  std::ifstream in(path);

  for (std::string lineraw; std::getline(in, lineraw);) {
    auto line = w2l::trim(lineraw);
    if (line.size() < 1) {
      continue;
    }
    auto segments = w2l::splitOnWhitespace(line, /* ignoreempty= */ false);

    // Vector of space-delimited words for transcript
    std::vector<std::string> transcriptWords;
    std::copy(
        segments.begin() + 3,
        segments.end(),
        std::back_inserter(transcriptWords));
    // Raw transcript
    auto transcript = w2l::join(" ", segments.begin() + 3, segments.end());
    output.insert({/* sid */ segments[0],
                   std::make_shared<Sample>(/* sid */ segments[0], /*
path */ segments[1], /* duration */ segments[2], transcript, transcriptWords)});
  }

  return output;
}

void writeTranscriptDictToFile(
    std::unordered_map<std::string, std::shared_ptr<Sample>> samples,
    const std::string& outpath) {
  if (outpath.empty()) {
    throw std::invalid_argument("Outpath to write list file to is empty");
  }
  if (w2l::fileExists(outpath)) {
    throw std::invalid_argument(
        "Output file at path '" + outpath + "' already exists.");
  }

  std::ofstream out(outpath);
  for (auto& sample : samples) {
    out << sample.second->toString() << std::endl;
  }
}

} // namespace dataset
} // namespace filter
