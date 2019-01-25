/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include <arrayfire.h>

#include "common/Defines.h"
#include "common/Dictionary.h"

namespace w2l {

typedef std::unordered_map<std::string, std::vector<std::vector<std::string>>>
    LexiconMap;

template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
  return vec;
}

// Converts an integer array to corresponding ascii representation.
// "-1"s are ignored.
template <>
inline std::vector<std::string> afToVector(const af::array& arr) {
  auto maxLen = arr.dims(0);
  auto batchSz = arr.dims(1);
  auto intVec = afToVector<int>(arr);

  std::vector<std::string> vec(batchSz);
  std::vector<char> charVec(maxLen);
  int curLen;
  for (int b = 0; b < batchSz; ++b) {
    auto offset = maxLen * b;
    for (curLen = 0; curLen < maxLen; ++curLen) {
      if (intVec[offset + curLen] == -1) {
        break;
      }
      charVec[curLen] = static_cast<char>(intVec[offset + curLen]);
    }
    vec[b] = std::string(charVec.begin(), charVec.begin() + curLen);
  }
  return vec;
}

template <typename T>
std::vector<T> afToVector(const fl::Variable& var) {
  return afToVector<T>(var.array());
}

std::string pathsConcat(const std::string& p1, const std::string& p2);

std::string trim(const std::string& str);

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl);

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty = false);

std::vector<std::string> split(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty = false);

bool dirExists(const std::string& path);

void dirCreate(const std::string& path);

bool fileExists(const std::string& path);

std::string getEnvVar(const std::string& key, const std::string& dflt = "");

std::string getCurrentDate();

std::string getCurrentTime();

template <
    typename FwdIt,
    typename = typename std::enable_if<std::is_same<
        typename std::decay<decltype(*std::declval<FwdIt>())>::type,
        std::string>::value>::type>
std::string join(const std::string& delim, FwdIt begin, FwdIt end);

std::string join(const std::string& delim, const std::vector<std::string>& vec);

std::string serializeGflags(const std::string& separator = "\n");

std::vector<std::string> getFileContent(const std::string& file);

bool startsWith(const std::string& input, const std::string& pattern);

template <class... Args>
std::string format(const char* fmt, Args&&... args);

std::vector<std::string> loadTarget(const std::string& filepath);

int64_t loadSize(const std::string& filepath);

Dictionary createTokenDict(const std::string& filepath);
Dictionary createTokenDict();

Dictionary createWordDict(const LexiconMap& lexicon);

int64_t numTotalParams(std::shared_ptr<fl::Module> module);

/************** Decoder helpers **************/
struct EmissionSet {
  std::vector<std::vector<float>> emissions;
  std::vector<std::vector<int>> wordTargets;
  std::vector<std::vector<int>> letterTargets;
  std::vector<std::string> sampleIds;
  std::vector<float> transition;
  std::vector<int> emissionT;
  int emissionN; // Assume alphabet size to be identical for all the samples

  std::string gflags; // Saving all the flags used in model training

  FL_SAVE_LOAD(
      emissions,
      wordTargets,
      letterTargets,
      sampleIds,
      transition,
      emissionT,
      emissionN,
      gflags)
};

LexiconMap loadWords(const std::string& fn, const int64_t maxNumWords);

std::vector<int> tokens2Tensor(const std::string&, const Dictionary&);

std::vector<int> tokens2Tensor(
    const std::vector<std::string>&,
    const Dictionary&);

std::string tensor2letters(const std::vector<int>&, const Dictionary&);

std::string tensor2words(const std::vector<int>&, const Dictionary&);

void validateTokens(std::vector<int>&, const int);

std::vector<int> tknTensor2wrdTensor(
    const std::vector<int>&,
    const Dictionary&,
    const Dictionary&,
    const int);

std::vector<int> wrdTensor2tknTensor(
    const std::vector<int>&,
    const Dictionary&,
    const Dictionary&,
    const int);

} // namespace w2l

#include "Utils-inl.h"
