/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/common/WordUtils.h"

#include <fstream>

#include "libraries/common/Defines.h"
#include "libraries/common/Utils.h"

namespace w2l {

Dictionary createWordDict(const LexiconMap& lexicon) {
  Dictionary dict;
  for (const auto& it : lexicon) {
    dict.addEntry(it.first);
  }
  dict.setDefaultIndex(dict.getIndex(kUnkToken));
  return dict;
}

LexiconMap loadWords(const std::string& filename, int maxWords) {
  LexiconMap lexicon;

  std::string line;
  std::ifstream infile(filename);

  if (!infile) {
    throw std::invalid_argument("Cannot open " + filename);
  }

  // Add at most `maxWords` words into the lexicon.
  // If `maxWords` is negative then no limit is applied.
  while (maxWords != lexicon.size() && std::getline(infile, line)) {
    // Parse the line into two strings: word and spelling.
    auto fields = splitOnWhitespace(line, true);
    if (fields.size() < 2) {
      throw std::runtime_error("[loadWords] Invalid line: " + line);
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
    const Dictionary& tokenDict,
    int maxReps) {
  std::vector<int> ret;
  ret.reserve(spelling.size());
  for (const auto& token : spelling) {
    ret.push_back(tokenDict.getIndex(token));
  }
  return packReplabels(ret, tokenDict, maxReps);
}

std::vector<int> validateIdx(std::vector<int> input, int unkIdx) {
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

std::vector<int> packReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps) {
  if (tokens.empty() || maxReps <= 0) {
    return tokens;
  }

  std::vector<int> replabelValueToIdx(maxReps + 1);
  for (int i = 1; i <= maxReps; ++i) {
    replabelValueToIdx[i] = dict.getIndex(std::to_string(i));
  }

  std::vector<int> result;
  int prevToken = -1;
  int numReps = 0;
  for (int token : tokens) {
    if (token == prevToken && numReps < maxReps) {
      numReps++;
    } else {
      if (numReps > 0) {
        result.push_back(replabelValueToIdx[numReps]);
        numReps = 0;
      }
      result.push_back(token);
      prevToken = token;
    }
  }
  if (numReps > 0) {
    result.push_back(replabelValueToIdx[numReps]);
  }
  return result;
}

std::vector<int> unpackReplabels(
    const std::vector<int>& tokens,
    const Dictionary& dict,
    int maxReps) {
  if (tokens.empty() || maxReps <= 0) {
    return tokens;
  }

  std::unordered_map<int, int> replabelIdxToValue;
  for (int i = 1; i <= maxReps; ++i) {
    replabelIdxToValue.emplace(dict.getIndex(std::to_string(i)), i);
  }

  std::vector<int> result;
  int prevToken = -1;
  for (int token : tokens) {
    auto it = replabelIdxToValue.find(token);
    if (it == replabelIdxToValue.end()) {
      result.push_back(token);
      prevToken = token;
    } else if (prevToken != -1) {
      result.insert(result.end(), it->second, prevToken);
      prevToken = -1;
    }
  }
  return result;
}

} // namespace w2l
