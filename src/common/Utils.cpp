/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "common/Utils.h"

#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "common/Defines.h"
#include "common/Transforms.h"

namespace w2l {

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
