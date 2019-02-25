/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include <glog/logging.h>

namespace w2l {

// Helper class used to send metadata about samples in a dataset for
// sorting/filtering
class SpeechSampleMetaInfo {
 private:
  double durationMs; // input audio length in ms
  int64_t refLength; // reference/target length
  int64_t idx; // index of the sample in the dataset

 public:
  SpeechSampleMetaInfo() {}

  SpeechSampleMetaInfo(double durationMs, int64_t refLength, int64_t i)
      : durationMs{durationMs}, refLength{refLength}, idx(i) {}

  double audiolength() const {
    return durationMs;
  }

  int64_t reflength() const {
    return refLength;
  }

  int64_t index() const {
    return idx;
  }
};

// Helper class used to store data in W2lListFilesDataset
class SpeechSample {
 private:
  std::string sampleId_; // utterance id
  std::string audioFilePath_; // full path to audio file
  std::vector<std::string> transcript_; // word transcripts

 public:
  SpeechSample() {}

  SpeechSample(
      std::string sampleId,
      std::string audioFile,
      std::vector<std::string> trans)
      : sampleId_(sampleId),
        audioFilePath_(audioFile),
        transcript_(std::move(trans)) {}

  std::string getSampleId() const {
    return sampleId_;
  }

  std::string getAudioFile() const {
    return audioFilePath_;
  }

  int64_t numWords() const {
    return transcript_.size();
  }

  std::vector<std::string> getTranscript() const {
    return transcript_;
  }

  std::string getTranscript(int64_t id) const {
    if (id >= transcript_.size()) {
      throw std::out_of_range("getTranscript idx out of range");
    }
    return transcript_[id];
  }
};

std::vector<int64_t> sortSamples(
    const std::vector<SpeechSampleMetaInfo>& samples,
    const std::string& dataorder,
    const int64_t inputbinsize,
    const int64_t outputbinsize);

void filterSamples(
    std::vector<SpeechSampleMetaInfo>& samples,
    const int64_t minInputSz,
    const int64_t maxInputSz,
    const int64_t minTargetSz,
    const int64_t maxTargetSz);

} // namespace w2l
