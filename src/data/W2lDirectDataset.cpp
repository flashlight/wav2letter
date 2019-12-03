#include <glog/logging.h>
#include <functional>
#include <numeric>
#include <string>

#include "common/Defines.h"
#include "data/W2lDirectDataset.h"
#include "data/HDF5Data.h"

namespace w2l {

W2lDirectDataset::W2lDirectDataset(
    const std::string& filename,
    const DictionaryMap& dicts,
    const LexiconMap& lexicon,
    int64_t batchSize /* = 1 */,
    int worldRank /* = 0 */,
    int worldSize /* = 1 */,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */,
    const std::string& rootdir /* = "" */)
    : W2lDataset(dicts, batchSize, worldRank, worldSize),
      lexicon_(lexicon),
      fallback2Ltr_(fallback2Ltr),
      skipUnk_(skipUnk) {
  includeWrd_ = (dicts.find(kWordIdx) != dicts.end());

  LOG_IF(FATAL, dicts.find(kTargetIdx) == dicts.end())
      << "Target dictionary does not exist";

  std::vector<SpeechSampleMetaInfo> speechSamplesMetaInfo;
  auto fileSampleInfo = loadListFile(filename);
  speechSamplesMetaInfo.insert(
      speechSamplesMetaInfo.end(),
      fileSampleInfo.begin(),
      fileSampleInfo.end());

  filterSamples(
      speechSamplesMetaInfo,
      FLAGS_minisz,
      FLAGS_maxisz,
      FLAGS_mintsz,
      FLAGS_maxtsz);
  sampleCount_ = speechSamplesMetaInfo.size();
  sampleSizeOrder_ = sortSamples(
      speechSamplesMetaInfo,
      FLAGS_dataorder,
      FLAGS_inputbinsize,
      FLAGS_outputbinsize);

  shuffle(-1);
  LOG(INFO) << "Total batches (i.e. iters): " << sampleBatches_.size();
}

W2lDirectDataset::~W2lDirectDataset() {
  threadpool_ = nullptr; // join all threads
}

std::vector<W2lLoaderData> W2lDirectDataset::getLoaderData(
    const int64_t idx) const {
  std::vector<W2lLoaderData> data(sampleBatches_[idx].size(), W2lLoaderData());
  for (int64_t id = 0; id < sampleBatches_[idx].size(); ++id) {
    auto i = sampleSizeOrder_[sampleBatches_[idx][id]];

    if (!(i >= 0 && i < data_.size())) {
      throw std::out_of_range(
          "W2lDirectDataset::getLoaderData idx out of range");
    }

    data[id].sampleId = data_[i].getSampleId();
    auto audioFile = data_[i].getAudioFile();

    if (FLAGS_wav2vec) {
      data[id].input = loadData(audioFile);
    }
    else {
      data[id].input = loadSound(audioFile);
    }

    data[id].targets[kTargetIdx] = wrd2Target(
        data_[i].getTranscript(),
        lexicon_,
        dicts_.at(kTargetIdx),
        fallback2Ltr_,
        skipUnk_);

    if (includeWrd_) {
      data[id].targets[kWordIdx] = data_[i].getTranscript();
    }
  }
  return data;
}

std::vector<float> W2lDirectDataset::loadSound(
    const std::string& audioHandle) const {
  return w2l::loadSound<float>(audioHandle);
}

std::vector<SpeechSampleMetaInfo> W2lDirectDataset::loadListFile(
    const std::string& filename) {

  std::vector<SpeechSampleMetaInfo> samplesMetaInfo;
  auto curDataSize = data_.size();
  int64_t idx = curDataSize;
  
  std::vector<std::string> transcript;
  data_.emplace_back(SpeechSample(
    "audiosmp",
    filename,
    transcript
  ));

  auto targets = wrd2Target(
      data_.back().getTranscript(),
      lexicon_,
      dicts_.at(kTargetIdx),
      fallback2Ltr_,
      skipUnk_);

  samplesMetaInfo.emplace_back(
      SpeechSampleMetaInfo(0, targets.size(), idx));

  ++idx;

  return samplesMetaInfo;
}
} // namespace w2l
