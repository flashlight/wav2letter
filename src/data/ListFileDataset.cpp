// Copyright 2004-present Facebook. All Rights Reserved.

#include "ListFileDataset.h"

#include "data/Sound.h"

namespace {
constexpr const size_t kIdIdx = 0;
constexpr const size_t kInIdx = 1;
constexpr const size_t kSzIdx = 2;
constexpr const size_t kTgtIdx = 3;
constexpr const size_t kNumCols = 4;

af::array toArray(const std::string& str) {
  return af::array(str.length(), str.data());
}
} // namespace

namespace w2l {

ListFileDataset::ListFileDataset(
    const std::string& filename,
    const DataTransformFunction& inFeatFunc /* = nullptr */,
    const DataTransformFunction& tgtFeatFunc /* = nullptr */)
    : inFeatFunc_(inFeatFunc), tgtFeatFunc_(tgtFeatFunc), numRows_(0) {
  std::ifstream inFile(filename);
  if (!inFile) {
    throw std::invalid_argument("Unable to open file -" + filename);
  }
  std::string line;
  while (std::getline(inFile, line)) {
    if (line.empty()) {
      continue;
    }
    auto splits = w2l::splitOnWhitespace(line, true);
    if (splits.size() < kNumCols) {
      throw std::runtime_error("Invalid line: " + line);
    }

    ids_.emplace_back(std::move(splits[kIdIdx]));
    inputs_.emplace_back(std::move(splits[kInIdx]));
    sizes_.emplace_back(std::stod(splits[kSzIdx]));
    targets_.emplace_back(w2l::join(
        " ", std::vector<std::string>(splits.begin() + kTgtIdx, splits.end())));
    ++numRows_;
  }
  inFile.close();
}

int64_t ListFileDataset::size() const {
  return numRows_;
}

std::vector<af::array> ListFileDataset::get(const int64_t idx) const {
  checkIndexBounds(idx);
  auto audio = loadAudio(inputs_[idx]);
  af::array input;
  if (inFeatFunc_) {
    input = inFeatFunc_(
        static_cast<void*>(audio.first.data()), audio.second, af::dtype::f32);
  } else {
    input = af::array(audio.second, audio.first.data());
  }
  af::array transcript = toArray(targets_[idx]);
  af::array target;
  if (tgtFeatFunc_) {
    std::vector<char> curTarget(targets_[idx].begin(), targets_[idx].end());
    target = tgtFeatFunc_(
        static_cast<void*>(curTarget.data()),
        {static_cast<dim_t>(curTarget.size())},
        af::dtype::b8);
  } else {
    target = transcript;
  }

  af::array sampleIdx = toArray(ids_[idx]);

  return {input, target, transcript, sampleIdx};
}

const std::vector<double>& ListFileDataset::getSampleSizes() const {
  return sizes_;
}

std::pair<std::vector<float>, af::dim4> ListFileDataset::loadAudio(
    const std::string& handle) const {
  auto info = w2l::loadSoundInfo(handle.c_str());
  return {w2l::loadSound<float>(handle.c_str()), {info.channels, info.frames}};
}

} // namespace w2l
