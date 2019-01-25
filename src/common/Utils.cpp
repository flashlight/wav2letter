/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Utils.h"

#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#include <array>
#include <fstream>
#include <functional>
#include <regex>

#include <flashlight/flashlight.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Transforms.h"

namespace {
const char* kSpaceChars = "\t\n\v\f\r ";
}

namespace w2l {

std::string pathsConcat(const std::string& p1, const std::string& p2) {
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  if (!p1.empty() && p1[p1.length() - 1] != sep) {
    return (trim(p1) + sep + trim(p2)); // Need to add a path separator
  } else {
    return (trim(p1) + trim(p2));
  }
}

std::string trim(const std::string& str) {
  auto i = str.find_first_not_of(kSpaceChars);
  if (i == std::string::npos) {
    return "";
  }
  auto j = str.find_last_not_of(kSpaceChars);
  DCHECK_NE(j, std::string::npos);
  DCHECK_LE(i, j);
  return str.substr(i, j - i + 1);
}

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl) {
  if (from.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), repl);
    pos += repl.length();
  }
}

namespace {
template <bool Any, typename Delim>
std::vector<std::string> splitImpl(
    const Delim& delim,
    std::string::size_type delimSize,
    const std::string& input,
    bool ignoreEmpty = false) {
  std::vector<std::string> result;
  std::string::size_type i = 0;
  while (true) {
    auto j = Any ? input.find_first_of(delim, i) : input.find(delim, i);
    if (j == std::string::npos) {
      break;
    }
    if (!(ignoreEmpty && i == j)) {
      result.emplace_back(input.begin() + i, input.begin() + j);
    }
    i = j + delimSize;
  }
  if (!(ignoreEmpty && i == input.size())) {
    result.emplace_back(input.begin() + i, input.end());
  }
  return result;
}
} // namespace

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty) {
  return splitImpl<false>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string>
split(const std::string& delim, const std::string& input, bool ignoreEmpty) {
  if (delim.empty()) {
    throw std::invalid_argument("delimiter is empty string");
  }
  return splitImpl<false>(delim, delim.size(), input, ignoreEmpty);
}

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty) {
  return splitImpl<true>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty) {
  return splitOnAnyOf(kSpaceChars, input, ignoreEmpty);
}

bool dirExists(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return false;
  } else if (info.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

void dirCreate(const std::string& path) {
  if (dirExists(path)) {
    return;
  }
  mode_t nMode = 0733;
  int nError = 0;
#ifdef _WIN32
  nError = _mkdir(path.c_str());
#else
  nError = mkdir(path.c_str(), nMode);
#endif
  if (nError != 0) {
    throw std::runtime_error(
        std::string() + "Unable to create directory - " + path);
  }
}

bool fileExists(const std::string& path) {
  std::ifstream fs(path, std::ifstream::in);
  return fs.good();
}

std::string getEnvVar(
    const std::string& key,
    const std::string& dflt /*= "" */) {
  char* val = getenv(key.c_str());
  return val ? std::string(val) : dflt;
}

std::string getCurrentDate() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  char buf[80];
  tstruct = localtime_r(&now, &tmbuf);

  strftime(buf, sizeof(buf), "%Y-%m-%d", tstruct);
  return buf;
}

std::string getCurrentTime() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  char buf[80];
  tstruct = localtime_r(&now, &tmbuf);

  strftime(buf, sizeof(buf), "%X", tstruct);
  return buf;
}

std::string join(
    const std::string& delim,
    const std::vector<std::string>& vec) {
  return join(delim, vec.begin(), vec.end());
}

std::string serializeGflags(const std::string& separator /* = "\n" */) {
  std::string serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
    serialized += "--" + itr->name + "=" + currVal + separator;
  }
  return serialized;
}

std::vector<std::string> loadTarget(const std::string& filepath) {
  std::vector<std::string> tokens;
  std::ifstream infile(filepath);
  if (!infile) {
    throw std::runtime_error(
        std::string() + "Could not read file '" + filepath + "'");
  }
  std::string line;
  while (std::getline(infile, line)) {
    auto tkns_str = splitOnWhitespace(line, true);
    for (const auto& tkn : tkns_str) {
      tokens.emplace_back(tkn);
    }
  }
  return tokens;
}

int64_t loadSize(const std::string& filepath) {
  std::string line;
  std::ifstream fs(filepath);
  std::getline(fs, line);
  return std::stol(line);
}

Dictionary createTokenDict(const std::string& filepath) {
  Dictionary dict;
  if (filepath.empty()) {
    LOG(FATAL) << "Empty filepath specified for token dictiinary.";
    return dict;
  }
  std::ifstream infile(trim(filepath));
  if (!infile) {
    LOG(FATAL) << "Unable to open dictionary file '" << filepath << "'";
  }
  std::string line;
  while (std::getline(infile, line)) {
    if (line.empty()) {
      continue;
    }
    auto tkns = splitOnWhitespace(line, true);
    auto idx = dict.indexSize();
    for (const auto& tkn : tkns) {
      dict.addToken(tkn, idx);
    }
  }
  LOG_IF(FATAL, !dict.isContiguous()) << "Invalid Dictionary! ";

  for (int64_t r = 1; r <= FLAGS_replabel; ++r) {
    dict.addToken(std::to_string(r));
  }
  // ctc expects the blank label last
  if (FLAGS_garbage || FLAGS_criterion == kCtcCriterion) {
    dict.addToken(kBlankToken);
  }
  if (FLAGS_eostoken) {
    dict.addToken(kEosToken);
  }
  return dict;
}

Dictionary createTokenDict() {
  return createTokenDict(pathsConcat(FLAGS_tokensdir, FLAGS_tokens));
}

Dictionary createWordDict(const LexiconMap& lexicon) {
  Dictionary dict;
  for (const auto& it : lexicon) {
    dict.addToken(it.first);
  }
  dict.setDefaultIndex(dict.getIndex(kUnkToken));
  return dict;
}

std::vector<std::string> getFileContent(const std::string& file) {
  std::vector<std::string> data;
  std::ifstream in(file);
  if (!in.good()) {
    throw std::runtime_error(
        std::string() + "Could not read file '" + file + "'");
  }
  std::string str;
  while (std::getline(in, str)) {
    data.emplace_back(str);
  }
  in.close();
  return data;
}

bool startsWith(const std::string& input, const std::string& pattern) {
  return (input.find(pattern) == 0);
}

int64_t numTotalParams(std::shared_ptr<fl::Module> module) {
  int64_t params = 0;
  for (auto& p : module->params()) {
    params += p.elements();
  }
  return params;
}

/************** Decoder helpers **************/
LexiconMap loadWords(const std::string& fn, const int64_t maxNumWords) {
  LexiconMap lexicon;

  std::string line;
  std::ifstream infile(fn);
  while (std::getline(infile, line)) {
    // Parse the line into two strings: word and spelling.
    auto fields = splitOnWhitespace(line, true);
    if (fields.size() < 2) {
      LOG(FATAL) << "[Loading words] Invalid line: " << line;
    }
    const std::string& word = fields[0];
    std::vector<std::string> spelling(fields.size() - 1);
    std::copy(fields.begin() + 1, fields.end(), spelling.begin());

    // Add the word into the dictionary.
    if (lexicon.find(word) == lexicon.end()) {
      lexicon[word] = {};
    }

    // Add the current spelling of the words to the list of spellings.
    lexicon[word].push_back(spelling);

    // Add at most maxn words into the dictionary.
    if (maxNumWords == lexicon.size()) {
      break;
    }
  }

  // Insert unknown word.
  lexicon[kUnkToken] = {};
  LOG(INFO) << "[Words] " << lexicon.size() << " tokens loaded.\n";
  return lexicon;
}

std::vector<int> tokens2Tensor(
    const std::string& spelling,
    const Dictionary& tokenDict) {
  std::vector<int> ret;
  ret.reserve(spelling.size());
  for (auto c : spelling) {
    ret.push_back(tokenDict.getIndex(std::string(1, c)));
  }
  replaceReplabels(ret, FLAGS_replabel, tokenDict);
  return ret;
}

std::vector<int> tokens2Tensor(
    const std::vector<std::string>& spelling,
    const Dictionary& tokenDict) {
  std::vector<int> ret;
  ret.reserve(spelling.size());
  for (auto c : spelling) {
    ret.push_back(tokenDict.getIndex(c));
  }
  replaceReplabels(ret, FLAGS_replabel, tokenDict);
  return ret;
}

std::string tensor2letters(
    const std::vector<int>& input,
    const Dictionary& tokenDict) {
  std::string ret = "";
  for (auto ltrIdx : input) {
    ret += tokenDict.getToken(ltrIdx);
  }
  return ret;
}

std::string tensor2words(
    const std::vector<int>& input,
    const Dictionary& wordDict) {
  std::string ret = "";
  for (auto wrdIdx : input) {
    ret += wordDict.getToken(wrdIdx) + " ";
  }
  return ret;
}

void validateTokens(std::vector<int>& input, const int unkIdx) {
  int newSize = 0;
  for (int i = 0; i < input.size(); i++) {
    if (input[i] >= 0 and input[i] != unkIdx) {
      input[newSize] = input[i];
      newSize++;
    }
  }
  input.resize(newSize);
}

std::vector<int> tknTensor2wrdTensor(
    const std::vector<int>& input,
    const Dictionary& wordDict,
    const Dictionary& tokenDict,
    const int spliterIdx) {
  std::vector<int> ret;
  std::string currentWord = "";
  for (auto ltrIdx : input) {
    if (ltrIdx == spliterIdx) {
      if (!currentWord.empty()) {
        if (wordDict.contains(currentWord)) {
          ret.push_back(wordDict.getIndex(currentWord));
        }
        currentWord = "";
      }
    } else {
      currentWord += tokenDict.getToken(ltrIdx);
    }
  }
  if (!currentWord.empty() && wordDict.contains(currentWord)) {
    ret.push_back(wordDict.getIndex(currentWord));
  }
  return ret;
}

std::vector<int> wrdTensor2tknTensor(
    const std::vector<int>& input,
    const Dictionary& wordDict,
    const Dictionary& tokenDict,
    const int spliterIdx) {
  std::vector<int> ret;
  int inputSize = input.size();
  for (int i = 0; i < inputSize; i++) {
    for (auto c : wordDict.getToken(input[i])) {
      ret.push_back(tokenDict.getIndex(std::string(1, c)));
    }
    if (i < inputSize - 1) {
      ret.push_back(spliterIdx);
    }
  }
  return ret;
}
} // namespace w2l
