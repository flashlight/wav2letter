/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "data/Utils.h"

namespace w2l {

std::vector<int64_t> sortSamples(
    const std::vector<SpeechSampleMetaInfo>& samples,
    const std::string& dataorder,
    const int64_t inputbinsize,
    const int64_t outputbinsize) {
  std::vector<int64_t> sortedIndices(samples.size());
  std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
  if (dataorder.compare("input_spiral") == 0) {
    // Sort samples in increasing order of output bins. For samples in the same
    // output bin, sorting is done based on input size in alternating manner
    // (spiral) for consecutive bins.
    VLOG(1) << "Doing data ordering by input_spiral";
    std::sort(
        sortedIndices.begin(),
        sortedIndices.end(),
        [&](int64_t i1, int64_t i2) {
          auto& s1 = samples[i1];
          auto& s2 = samples[i2];
          auto s1_y = s1.reflength() / outputbinsize;
          auto s2_y = s2.reflength() / outputbinsize;
          if (s1_y != s2_y) {
            return s1_y < s2_y;
          }
          if (s1_y % 2 == 0) {
            return s1.audiolength() < s2.audiolength();
          } else {
            return s2.audiolength() < s1.audiolength();
          }
        });
  } else if (dataorder.compare("output_spiral") == 0) {
    // Sort samples in increasing order of input bins. For samples in the same
    // input bin, sorting is done based on output size in alternating manner
    // (spiral) for consecutive bins.
    VLOG(1) << "Doing data ordering by output_spiral";
    std::sort(
        sortedIndices.begin(),
        sortedIndices.end(),
        [&](int64_t i1, int64_t i2) {
          auto& s1 = samples[i1];
          auto& s2 = samples[i2];
          int s1_x = s1.audiolength() / inputbinsize;
          int s2_x = s2.audiolength() / inputbinsize;
          if (s1_x != s2_x) {
            return s1_x < s2_x;
          }
          if (s1_x % 2 == 0) {
            return s1.reflength() < s2.reflength();
          } else {
            return s2.reflength() < s1.reflength();
          }
        });
  } else if (dataorder.compare("input") == 0) {
    // Sort by input size
    VLOG(1) << "Doing data ordering by input";
    std::sort(
        sortedIndices.begin(),
        sortedIndices.end(),
        [&](int64_t i1, int64_t i2) {
          auto& s1 = samples[i1];
          auto& s2 = samples[i2];
          return s1.audiolength() < s2.audiolength();
        });
  } // Default is no sorting.

  std::vector<int64_t> sortedSampleIndices(samples.size());
  for (size_t i = 0; i < sortedSampleIndices.size(); ++i) {
    sortedSampleIndices[i] = samples[sortedIndices[i]].index();
  }
  return sortedSampleIndices;
}
void filterSamples(
    std::vector<SpeechSampleMetaInfo>& samples,
    const int64_t minInputSz,
    const int64_t maxInputSz,
    const int64_t minTargetSz,
    const int64_t maxTargetSz) {
  auto initialSize = samples.size();
  samples.erase(
      std::remove_if(
          samples.begin(),
          samples.end(),
          [minInputSz, maxInputSz, minTargetSz, maxTargetSz](
              const SpeechSampleMetaInfo& sample) {
            return sample.audiolength() < minInputSz ||
                sample.audiolength() > maxInputSz ||
                sample.reflength() < minTargetSz ||
                sample.reflength() > maxTargetSz;
          }),
      samples.end());
  LOG(INFO) << "Filtered " << initialSize - samples.size() << "/" << initialSize
            << " samples";
}
} // namespace w2l
