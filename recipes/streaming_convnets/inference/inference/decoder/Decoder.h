/*
 * Copyright (c) 2018-present, Facebook, Inc.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

namespace w2l {
namespace streaming {

struct WordUnit {
  // The decoded word.
  std::string word;
  // Time frame in hypothesis of the first letter of the decoded word.
  int beginTimeFrame;
  // Time frame in hypothesis of the decoded word.
  int endTimeFrame;

  WordUnit(const std::string& word, int beginTime, int endTime)
      : word(word), beginTimeFrame(beginTime), endTimeFrame(endTime) {}
};

class Decoder;

// Implementation of Decoder Factory that loads and initializes common decoder
// parameters that are shared across different Decoder instances. This class
// is thread safe and supports creation of Decoder instances that can be used
// to decode streams.
class DecoderFactory {
 public:
  // Loads all the parameters and initializes decoder model.
  DecoderFactory(
      const std::string& letterDictFile,
      const std::string& wordDictFile,
      const std::string& languageModelFile,
      const std::vector<float>& transitions,
      fl::lib::text::SmearingMode smearing,
      const std::string& silenceToken,
      const int repetitionLabel);

  // Creates provided Decoder instance with specified options and allocator.
  // The Decoder instance uses provided allocator to manage its memory.
  Decoder createDecoder(
      const fl::lib::text::LexiconDecoderOptions& options) const;

  // Parse the raw decoder results and form a list of WordUnit
  std::vector<WordUnit> result2Words(
      const fl::lib::text::DecodeResult& result) const;

  // Returns size of the alphabet (=dimension of transitions matrix).
  size_t alphabetSize() const;

 private:
  fl::lib::text::Dictionary wordMap_;
  fl::lib::text::Dictionary letterMap_;
  size_t alphabetSize_;
  int silence_;
  int blank_;
  int unk_;
  int repetitionLabel_;
  fl::lib::text::LMPtr lm_;
  fl::lib::text::TriePtr trie_;
  std::vector<float> transitions_;

  // Helper functions to unpack RepLabels from the transcription
  bool isInt(const std::string& s) const;
  int nUTF8Bytes(const unsigned char& c) const;
  std::string unpackReplabels(const std::string& input) const;
};

// This is a thin wrapper over wav2letter decoder to wav2letter online inference
// pipline.
class Decoder {
 public:
  Decoder() {}

  Decoder(
      const DecoderFactory* factory,
      std::shared_ptr<fl::lib::text::Decoder> decoder)
      : factory_(std::make_shared<DecoderFactory>(*factory)),
        decoder_(decoder) {}

  void start();

  void run(const float* input, size_t size);

  void finish();

  std::vector<WordUnit> getBestHypothesisInWords(int lookBack) const;

  /* Prune the hypothesis space */
  void prune(int lookBack = 0);

 private:
  const std::shared_ptr<DecoderFactory> factory_;
  std::shared_ptr<fl::lib::text::Decoder> decoder_;
};

} // namespace streaming
} // namespace w2l
