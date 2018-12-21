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
#include <vector>

#include <arrayfire.h>

#include "common/Defines.h"
#include "common/Dictionary.h"

namespace w2l {

template <typename T>
std::vector<T> afToVector(const af::array& arr) {
  std::vector<T> vec(arr.elements());
  arr.host(vec.data());
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

Dictionary makeDictionary(const std::string& filepath);
Dictionary makeDictionary();

int64_t numTotalParams(std::shared_ptr<fl::Module> module);

/************** Decoder helpers **************/
struct EmissionSet {
  std::vector<std::vector<float>> emissions;
  std::vector<std::vector<int>> wordTargets;
  std::vector<std::vector<int>> letterTargets;
  std::vector<float> transition;
  std::vector<int> emissionT;
  int emissionN;

  FL_SAVE_LOAD(
      emissions,
      wordTargets,
      letterTargets,
      transition,
      emissionT,
      emissionN)
};

typedef std::unordered_map<std::string, std::vector<std::vector<std::string>>>
    Word2Spell;

Word2Spell loadWords(const std::string& fn, const int64_t maxNumWords);

std::vector<int> spelling2tensor(const std::string&, const Dictionary&);

std::vector<int> spelling2tensor(
    const std::vector<std::string>&,
    const Dictionary&);

std::string tensor2letters(const std::vector<int>&, const Dictionary&);

std::string tensor2words(const std::vector<int>&, const Dictionary&);

void validateWords(std::vector<int>&, const int);

std::vector<int> ltrTensor2wrdTensor(
    const std::vector<int>&,
    const Dictionary&,
    const Dictionary&,
    const int);

std::vector<int> wrdTensor2ltrTensor(
    const std::vector<int>&,
    const Dictionary&,
    const Dictionary&,
    const int);
} // namespace w2l

#include "Utils-inl.h"
