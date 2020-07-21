/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <codecvt>
#include <locale>
#include "common/FlashlightUtils.h"
#include "common/Transforms.h"
namespace w2l {

namespace alignment {

struct AlignedWord {
  std::string word;
  double startTimeMs;
  double endTimeMs;
};

void remapUTFWord(std::u16string& input, int replabel) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;

  // dedup labels
  auto it = std::unique(input.begin(), input.end());
  input.resize(std::distance(input.begin(), it));

  // map of replabels
  std::unordered_map<char16_t, int64_t> replabelMap;
  for (int64_t i = 1; i <= replabel; ++i) {
    replabelMap[utf16conv.from_bytes(std::to_string(i))[0]] = i;
  }

  std::u16string output;
  output += input[0];
  for (size_t i = 1; i < input.size(); ++i) {
    auto repCount = replabelMap.find(input[i]);
    if (repCount != replabelMap.end()) {
      for (auto j = 0; j < repCount->second; j++) {
        output += input[i - 1];
      }
    } else {
      output += input[i];
    }
  }

  std::swap(input, output);
}

std::vector<std::vector<std::string>> mapIndexToToken(
    af::array paths,
    w2l::DictionaryMap dicts) {
  const int B = paths.dims(1);
  const int T = paths.dims(0);
  std::vector<std::vector<std::string>> batchTokensPath;
  for (int b = 0; b < B; b++) {
    std::vector<std::string> tokens;
    for (int t = 0; t < T; t++) {
      int p = paths(t, b).scalar<int>();
      if (p == -1) {
        break;
      }
      auto token = dicts[kTargetIdx].getEntry(p);
      tokens.push_back(token);
    }
    batchTokensPath.push_back(tokens);
  }
  return batchTokensPath;
}

std::vector<AlignedWord> postprocessCTC(
    const std::vector<std::string>& ltrs,
    int replabel,
    double msPerFrame) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
  std::vector<std::u16string> utf16Ltrs;
  for (const std::string& l : ltrs) {
    utf16Ltrs.push_back(utf16conv.from_bytes(l));
  }

  std::vector<AlignedWord> alignedWords;
  const int T = utf16Ltrs.size();
  int i = 0;

  while (i < T) {
    std::u16string currWord = utf16Ltrs[i];
    int j = i;
    while (j + 1 < T && utf16Ltrs[i] == utf16Ltrs[j + 1]) {
      j += 1;
    }
    j++;
    double endTimeMs = msPerFrame * (j);
    double startTimeMs = msPerFrame * i;
    remapUTFWord(currWord, replabel);
    alignedWords.push_back(
        AlignedWord{utf16conv.to_bytes(currWord), startTimeMs, endTimeMs});
    i = j;
  }
  return alignedWords;
}

std::vector<AlignedWord> postprocessASG(
    const std::vector<std::string>& ltrs,
    int replabel,
    double msPerFrame) {
  std::wstring_convert<std::codecvt_utf8_utf16<char16_t>, char16_t> utf16conv;
  std::vector<std::u16string> utf16Ltrs;
  for (const std::string& l : ltrs) {
    utf16Ltrs.push_back(utf16conv.from_bytes(l));
  }

  std::vector<AlignedWord> alignedWords;
  std::u16string currWord;
  int stFrame = 0;
  bool inWord = false;
  int silStart = 0;
  for (int i = 0; i < utf16Ltrs.size(); i++) {
    if (utf16Ltrs[i] == utf16conv.from_bytes(w2l::kSilToken)) {
      if (inWord) { // found end of word, insert
        const double endTimeMs = msPerFrame * i;
        const double startTimeMs = msPerFrame * stFrame;
        remapUTFWord(currWord, replabel);
        alignedWords.push_back(
            {utf16conv.to_bytes(currWord), startTimeMs, endTimeMs});
        inWord = false;
        silStart = i;
      }
    } else if (!inWord) { // starting new word
      stFrame = i;
      currWord = utf16Ltrs[i];
      inWord = true;
      // Also insert silence
      if (silStart < i - 1) {
        alignedWords.push_back({"$", msPerFrame * silStart, msPerFrame * (i)});
      }
    } else { // continue in same word
      currWord += utf16Ltrs[i];
    }
  }

  // Take care of trailing silence or trailing word
  const double endTimeMs = msPerFrame * (utf16Ltrs.size());
  if (inWord) {
    // we may encounter trailing word only
    // if we train without -surround='|'
    currWord += utf16Ltrs[utf16Ltrs.size() - 1];
    remapUTFWord(currWord, replabel);
    alignedWords.push_back(
        {utf16conv.to_bytes(currWord), msPerFrame * stFrame, endTimeMs});
  } else {
    alignedWords.push_back({"$", msPerFrame * silStart, endTimeMs});
  }
  return alignedWords;
  return std::vector<AlignedWord>();
}

// TODO T61657501 move the "wordSegmenter" function into the criterion class
// once we are ready to move out of experimental
std::function<
    std::vector<AlignedWord>(const std::vector<std::string>&, int, double)>
getWordSegmenter(std::shared_ptr<w2l::SequenceCriterion> criterion) {
  const std::string& desc = criterion->prettyString();
  const std::string asgDesc = "AutoSegmentationCriterion";
  const std::string ctcDesc = "ConnectionistTemporalClassificationCriterion";
  if (desc == ctcDesc) {
    return postprocessCTC;
  } else if (desc == asgDesc) {
    return postprocessASG;
  } else {
    throw std::invalid_argument("Alignment not supported for this criterion");
    return std::function<std::vector<AlignedWord>(
        const std::vector<std::string>&, int, double)>();
  }
}

// Utility function which converts the aligned words
// into CTM format which is compatible with AML's alignment output.
// this format can be used in several workflows later, including
// segmentation workflow.
std::string getCTMFormat(std::vector<AlignedWord> alignedWords) {
  std::stringstream ctmString;
  int i = 0;
  for (auto& alignedWord : alignedWords) {
    double stTimeSec = alignedWord.startTimeMs / 1000.0;
    double durationSec =
        (alignedWord.endTimeMs - alignedWord.startTimeMs) / 1000.0;
    ctmString << "ID A " << stTimeSec << " " << durationSec << " "
              << alignedWord.word;
    if (i < alignedWords.size() - 1) {
      ctmString << "\\n";
    }
    i++;
  }
  return ctmString.str();
}

} // namespace alignment
} // namespace w2l
