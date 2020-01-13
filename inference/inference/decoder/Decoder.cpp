/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "libraries/common/Defines.h"
#include "libraries/common/WordUtils.h"
#include "libraries/decoder/LexiconDecoder.h"
#include "libraries/decoder/LexiconFreeDecoder.h"
#include "libraries/lm/KenLM.h"
#include "libraries/lm/ZeroLM.h"

#include "inference/decoder/Decoder.h"

namespace w2l {
namespace streaming {

/* ===================== DecoderFactory ===================== */
DecoderFactory::DecoderFactory(
    const std::string& letterDictFile,
    const std::string& wordDictFile,
    const std::string& languageModelFile,
    const std::vector<float>& transitions,
    SmearingMode smearing,
    const std::string& silenceToken,
    const int repetitionLabel)
    : letterMap_(Dictionary(letterDictFile)),
      alphabetSize_(letterMap_.indexSize()),
      repetitionLabel_(repetitionLabel),
      transitions_(transitions) {
  /* 1. Load letter dictionary */
  if (alphabetSize_ == 0) {
    throw std::invalid_argument("Invalid letter dictionary.");
  }
  std::cerr << "[Letters] " << alphabetSize_ << " tokens loaded.\n";
  silence_ = letterMap_.getIndex(silenceToken);
  blank_ =
      letterMap_.contains(kBlankToken) ? letterMap_.getIndex(kBlankToken) : -1;

  /* 2. Load word dictionary */
  LexiconMap lexicon;
  unk_ = -1;
  if (!wordDictFile.empty()) {
    lexicon = loadWords(wordDictFile);
    wordMap_ = createWordDict(lexicon);
    int nWords = wordMap_.indexSize();
    if (nWords == 0) {
      throw std::invalid_argument("Invalid word dictionary.");
    }
    std::cerr << "[Words] " << nWords << " words loaded.\n";
    unk_ = wordMap_.getIndex(kUnkToken);
  }

  /* 3. Load language model. */
  if (!languageModelFile.empty()) {
    lm_ = std::make_shared<KenLM>(languageModelFile.c_str(), wordMap_);
    if (!lm_) {
      throw std::invalid_argument("Could not load LM.");
    }
  } else {
    lm_ = std::make_shared<ZeroLM>();
  }

  /* 4. Plant trie */
  if (!wordDictFile.empty()) {
    // Init Trie.
    trie_ = std::make_shared<Trie>(alphabetSize_, silence_);
    auto startState = lm_->start(false);
    for (const auto& it : lexicon) {
      const std::string& word = it.first;
      int usrIdx = wordMap_.getIndex(word);
      float score = -1;
      LMStatePtr dummyState;
      std::tie(dummyState, score) = lm_->score(startState, usrIdx);

      for (const auto& tokens : it.second) {
        auto tokensTensor = tkn2Idx(tokens, letterMap_, repetitionLabel_);
        trie_->insert(tokensTensor, usrIdx, score);
      }
    }

    // Smearing.
    trie_->smear(smearing);
  }
}

Decoder DecoderFactory::createDecoder(const DecoderOptions& opt) const {
  if (trie_) {
    auto decoder = std::make_shared<LexiconDecoder>(
        opt, trie_, lm_, silence_, blank_, unk_, transitions_, false);
    std::cerr << "Creating LexiconDecoder instance.\n";
    return Decoder(this, decoder);
  } else {
    auto decoder = std::make_shared<LexiconFreeDecoder>(
        opt, lm_, silence_, blank_, transitions_);
    std::cerr << "Creating LexiconFreeDecoder instance.\n";
    return Decoder(this, decoder);
  }
}

size_t DecoderFactory::alphabetSize() const {
  return alphabetSize_;
}

std::vector<WordUnit> DecoderFactory::result2Words(
    const DecodeResult& result) const {
  int seqLength = result.tokens.size();
  if (seqLength == 0) {
    return std::vector<WordUnit>{};
  }

  std::vector<WordUnit> wordPrediction;
  // If result.words is meaningful
  if (trie_) {
    int beginTime;
    bool tracking = false;
    for (int i = 0; i < seqLength; i++) {
      // We are not seeing meanful words yet
      if (!tracking && result.tokens[i] != silence_ &&
          result.tokens[i] != blank_) {
        beginTime = i;
        tracking = true;
      }
      // We are tracking a valid word
      if (tracking && result.words[i] > 0) {
        wordPrediction.emplace_back(
            wordMap_.getEntry(result.words[i]), beginTime, i);
        tracking = false;
      }
    }
  }
  // Else, parse from the token sequence
  else {
    int beginTime;
    std::string curWord = "";
    bool prevBlank = false;
    for (int i = 0; i < seqLength; i++) {
      bool curBlank = result.tokens[i] == blank_;
      if (result.tokens[i] != silence_ && !curBlank) {
        beginTime = curWord.empty() ? i : beginTime;
        if (prevBlank || i == 0 || result.tokens[i - 1] != result.tokens[i]) {
          curWord += letterMap_.getEntry(result.tokens[i]);
        }
      }
      if (result.tokens[i] == silence_ || i == seqLength - 1) {
        if (!curWord.empty()) {
          wordPrediction.emplace_back(unpackReplabels(curWord), beginTime, i);
          curWord = "";
        }
      }
      prevBlank = curBlank;
    }
  }

  return wordPrediction;
}

bool DecoderFactory::isInt(const std::string& s) const {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) {
    ++it;
  }
  return !s.empty() && it == s.end();
}

int DecoderFactory::nUTF8Bytes(const unsigned char& c) const {
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
  return curTknBytes;
}

std::string DecoderFactory::unpackReplabels(const std::string& input) const {
  std::string output = "";
  std::string prevTkn = "";

  const int len = input.length();
  for (int i = 0; i < len;) {
    const unsigned char c = (unsigned char)(input[i]);
    int curTknBytes = nUTF8Bytes(c);
    if (curTknBytes == -1 || i + curTknBytes > len) {
      throw std::runtime_error(
          "unpackReplabels() failed due to invalid UTF-8 : " + input);
    }

    int rep = -1;
    const std::string curToken(
        input.begin() + i, input.begin() + i + curTknBytes);
    if (isInt(curToken)) {
      rep = std::stoi(curToken);
    }

    // Replabel
    if (rep > 0 && rep <= repetitionLabel_) {
      for (int j = 0; j < rep; j++) {
        output += prevTkn;
      }
      prevTkn = "";
    }
    // Normal case
    else {
      output += curToken;
      prevTkn = curToken;
    }
    i += curTknBytes;
  }

  return output;
}

/* ===================== Decoder===================== */

void Decoder::start() {
  decoder_->decodeBegin();
}

void Decoder::run(const float* input, size_t size) {
  if (!input) {
    throw std::invalid_argument(
        "Decoder::run(input=nullptr, size=" + std::to_string(size) +
        ") empty input.");
  }
  const int N = static_cast<int>(factory_->alphabetSize());
  if (size % N != 0) {
    throw std::invalid_argument(
        "size must be devisible in alphabet size in Decoder::run(input=" +
        std::to_string(size) + ", size=" + std::to_string(size) +
        ") alphabet size=" + std::to_string(N));
  }
  const int T = size / factory_->alphabetSize();
  decoder_->decodeStep(input, T, N);
}

void Decoder::finish() {
  decoder_->decodeEnd();
}

std::vector<WordUnit> Decoder::getBestHypothesisInWords(int lookBack) const {
  DecodeResult rawResult = decoder_->getBestHypothesis(lookBack);
  return factory_->result2Words(rawResult);
}

void Decoder::prune(int lookBack) {
  decoder_->prune(lookBack);
}

} // namespace streaming
} // namespace w2l
