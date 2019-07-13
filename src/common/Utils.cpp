/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/Utils.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>
#include <iostream>

#include "common/Defines.h"
#include "common/Transforms.h"

static constexpr const char* kSpaceChars = "\t\n\v\f\r ";

namespace w2l {

std::string trim(const std::string& str) {
  auto i = str.find_first_not_of(kSpaceChars);
  if (i == std::string::npos) {
    return "";
  }
  auto j = str.find_last_not_of(kSpaceChars);
  if (j == std::string::npos || i > j) {
    return "";
  }
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

bool startsWith(const std::string& input, const std::string& pattern) {
  return (input.find(pattern) == 0);
}

template <bool Any, typename Delim>
static std::vector<std::string> splitImpl(
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

std::string join(
    const std::string& delim,
    const std::vector<std::string>& vec) {
  return join(delim, vec.begin(), vec.end());
}

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
  mode_t nMode = 0755;
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

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  auto lit = lexicon.find(word);
  if (lit != lexicon.end()) {
    if (lit->second.size() > 1 &&
        FLAGS_sampletarget >
            static_cast<float>(std::rand()) / static_cast<float>(RAND_MAX)) {
      return lit->second[std::rand() % lit->second.size()];
    } else {
      return lit->second[0];
    }
  }

  std::vector<std::string> res;
  if (fallback2Ltr) {
    std::cerr
        << "Falling back to using letters as targets for the unknown word: "
        << word << "\n";
    auto tokens = splitWrd(word);
    for (const auto& tkn : tokens) {
      if (dict.contains(tkn)) {
        res.push_back(tkn);
      } else if (skipUnk) {
        std::cerr
            << "Skipping unknown token '" << tkn
            << "' when falling back to letter target for the unknown word: "
            << word << "\n";
      } else {
        throw std::invalid_argument(
            "Unknown token '" + tkn +
            "' when falling back to letter target for the unknown word: " +
            word);
      }
    }
  } else if (skipUnk) {
    std::cerr << "Skipping unknown word '" << word
              << "' when generating target\n";
  } else {
    throw std::invalid_argument("Unknown word in the lexicon: " + word);
  }
  return res;
}

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr /* = false */,
    bool skipUnk /* = false */) {
  std::vector<std::string> res;
  for (auto w : words) {
    auto t = wrd2Target(w, lexicon, dict, fallback2Ltr, skipUnk);

    if (t.size() == 0) {
      continue;
    }

    // remove duplicate word separators in the beginning of each target token
    if (res.size() > 0 && !FLAGS_wordseparator.empty() &&
        t[0].length() >= FLAGS_wordseparator.length() &&
        t[0].compare(0, FLAGS_wordseparator.length(), FLAGS_wordseparator) ==
            0) {
      res.pop_back();
    }

    res.insert(res.end(), t.begin(), t.end());

    if (!FLAGS_wordseparator.empty() &&
        !(res.back().length() >= FLAGS_wordseparator.length() &&
          res.back().compare(
              res.back().length() - FLAGS_wordseparator.length(),
              FLAGS_wordseparator.length(),
              FLAGS_wordseparator) == 0)) {
      res.emplace_back(FLAGS_wordseparator);
    }
  }

  if (res.size() > 0 && res.back() == FLAGS_wordseparator) {
    res.pop_back();
  }
  return res;
}

/************** Decoder helpers **************/

Dictionary createWordDict(const LexiconMap& lexicon) {
  Dictionary dict;
  for (const auto& it : lexicon) {
    dict.addEntry(it.first);
  }
  dict.setDefaultIndex(dict.getIndex(kUnkToken));
  return dict;
}

LexiconMap loadWords(const std::string& fn, const int64_t maxNumWords) {
  LexiconMap lexicon;

  std::string line;
  std::ifstream infile(fn);

  if (!infile) {
    throw std::invalid_argument("Cannot open " + fn);
  }

  while (std::getline(infile, line)) {
    // Parse the line into two strings: word and spelling.
    auto fields = splitOnWhitespace(line, true);
    if (fields.size() < 2) {
      throw std::runtime_error("[Loading words] Invalid line: " + line);
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
  return lexicon;
}

std::vector<std::string> splitWrd(const std::string& word) {
  std::vector<std::string> tokens;
  tokens.reserve(word.size());
  int len = word.length();
  for (int i = 0; i < len;) {
    auto c = static_cast<unsigned char>(word[i]);
    int curTknBytes = -1;
    // UTF-8 checks, works for ASCII automatically
    if ((c & 0x80) == 0) {
      curTknBytes = 1;
    } else if ((c & 0xE0) == 0xC0) {
      curTknBytes = 2;
    } else if ((c & 0xF0) == 0xE0) {
      curTknBytes = 3;
    } else if ((c & 0xF8) == 0xF0) {
      curTknBytes = 4;
    }
    if (curTknBytes == -1 || i + curTknBytes > len) {
      throw std::runtime_error("splitWrd: invalid UTF-8 : " + word);
    }
    tokens.emplace_back(word.begin() + i, word.begin() + i + curTknBytes);
    i += curTknBytes;
  }
  return tokens;
}

std::vector<int> tkn2Idx(
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

std::vector<int> validateIdx(std::vector<int> input, const int unkIdx) {
  int newSize = 0;
  for (int i = 0; i < input.size(); i++) {
    if (input[i] >= 0 and input[i] != unkIdx) {
      input[newSize] = input[i];
      newSize++;
    }
  }
  input.resize(newSize);

  return input;
}

std::vector<std::string> tknIdx2Ltr(
    const std::vector<int>& labels,
    const Dictionary& d) {
  std::vector<std::string> result;

  for (auto id : labels) {
    auto token = d.getEntry(id);
    if (FLAGS_usewordpiece) {
      auto splitToken = splitWrd(token);
      for (const auto& c : splitToken) {
        result.emplace_back(c);
      }
    } else {
      result.emplace_back(token);
    }
  }

  if (result.size() > 0 && !FLAGS_wordseparator.empty()) {
    if (result.front() == FLAGS_wordseparator) {
      result.erase(result.begin());
    }
    if (!result.empty() && result.back() == FLAGS_wordseparator) {
      result.pop_back();
    }
  }

  return result;
}

std::vector<std::string> tkn2Wrd(const std::vector<std::string>& input) {
  std::vector<std::string> words;
  std::string currentWord = "";
  for (auto& tkn : input) {
    if (tkn == FLAGS_wordseparator) {
      if (!currentWord.empty()) {
        words.push_back(currentWord);
        currentWord = "";
      }
    } else {
      currentWord += tkn;
    }
  }
  if (!currentWord.empty()) {
    words.push_back(currentWord);
  }
  return words;
}

std::vector<std::string> wrdIdx2Wrd(
    const std::vector<int>& input,
    const Dictionary& wordDict) {
  std::vector<std::string> words;
  for (auto wrdIdx : input) {
    words.push_back(wordDict.getEntry(wrdIdx));
  }
  return words;
}

std::vector<std::string> tknTarget2Ltr(
    std::vector<int> tokens,
    const Dictionary& tokenDict) {
  if (tokens.empty()) {
    return std::vector<std::string>{};
  }

  if (FLAGS_criterion == kSeq2SeqCriterion) {
    if (tokens.back() == tokenDict.getIndex(kEosToken)) {
      tokens.pop_back();
    }
  }
  remapLabels(tokens, tokenDict);

  return tknIdx2Ltr(tokens, tokenDict);
}

std::vector<std::string> tknPrediction2Ltr(
    std::vector<int> tokens,
    const Dictionary& tokenDict) {
  if (tokens.empty()) {
    return std::vector<std::string>{};
  }

  if (FLAGS_criterion == kCtcCriterion || FLAGS_criterion == kAsgCriterion) {
    uniq(tokens);
  }
  if (FLAGS_criterion == kCtcCriterion) {
    int blankIdx = tokenDict.getIndex(kBlankToken);
    tokens.erase(
        std::remove(tokens.begin(), tokens.end(), blankIdx), tokens.end());
  }
  tokens = validateIdx(tokens, -1);
  remapLabels(tokens, tokenDict);

  return tknIdx2Ltr(tokens, tokenDict);
}

} // namespace w2l
