/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/examples/AudioToWords.h"

#include <fstream>
#include <functional>

#include "inference/common/IOBuffer.h"
#include "inference/examples/Util.h"

namespace w2l {
namespace streaming {

namespace {

void printChunckTranscription(
    std::ostream& output,
    const std::vector<WordUnit>& wordUnits,
    int chunckStartTime,
    int chunckEndTime) {
  output << chunckStartTime << "," << chunckEndTime << ",";
  for (const auto& wordUnit : wordUnits) {
    output << wordUnit.word << " ";
  }
  output << std::endl;
}

} // namespace

void audioStreamToWordsStream(
    std::istream& inputAudioStream,
    std::ostream& outputWordsStream,
    std::shared_ptr<Sequential> dnnModule,
    std::shared_ptr<const DecoderFactory> decoderFactory,
    const DecoderOptions& decoderOptions,
    int nTokens) {
  constexpr const int lookBack = 0;
  constexpr const size_t kWavHeaderNumBytes = 44;
  constexpr const float kMaxUint16 = static_cast<float>(0x8000);
  constexpr const int kAudioWavSamplingFrequency = 16000; // 16KHz audio.
  constexpr const int kChunkSizeMsec = 500;

  auto decoder = decoderFactory->createDecoder(decoderOptions);

  inputAudioStream.ignore(kWavHeaderNumBytes);

  const int minChunkSize = kChunkSizeMsec * kAudioWavSamplingFrequency / 1000;
  auto input = std::make_shared<streaming::ModuleProcessingState>(1);
  auto inputBuffer = input->buffer(0);
  int audioSampleCount = 0;

  // The same output object is returned by start(), run() and finish()
  auto output = dnnModule->start(input);
  auto outputBuffer = output->buffer(0);
  decoder.start();
  bool finish = false;

  outputWordsStream << "#start (msec), end(msec), transcription" << std::endl;
  while (!finish) {
    int curChunkSize = readTransformStreamIntoBuffer<int16_t, float>(
        inputAudioStream, inputBuffer, minChunkSize, [](int16_t i) -> float {
          return static_cast<float>(i) / kMaxUint16;
        });

    if (curChunkSize >= minChunkSize) {
      dnnModule->run(input);
      float* data = outputBuffer->data<float>();
      int size = outputBuffer->size<float>();
      if (data && size > 0) {
        decoder.run(data, size);
      }
    } else {
      dnnModule->finish(input);
      float* data = outputBuffer->data<float>();
      int size = outputBuffer->size<float>();
      if (data && size > 0) {
        decoder.run(data, size);
      }
      decoder.finish();
      finish = true;
    }

    /* Print results */
    const int chunk_start_ms =
        (audioSampleCount / (kAudioWavSamplingFrequency / 1000));
    const int chunk_end_ms =
        ((audioSampleCount + curChunkSize) /
         (kAudioWavSamplingFrequency / 1000));
    printChunckTranscription(
        outputWordsStream,
        decoder.getBestHypothesisInWords(lookBack),
        chunk_start_ms,
        chunk_end_ms);
    audioSampleCount += curChunkSize;

    // Consume and prune
    const int nFramesOut = outputBuffer->size<float>() / nTokens;
    outputBuffer->consume<float>(nFramesOut * nTokens);
    decoder.prune(lookBack);
  }
}

namespace {

void audioFileToWordsFileImpl(
    const std::string& inputFileName,
    const std::string& outputFileName,
    std::shared_ptr<streaming::Sequential> dnnModule,
    std::shared_ptr<const DecoderFactory> decoderFactory,
    const DecoderOptions& decoderOptions,
    int nTokens,
    std::ostream* errorStream) {
  std::ifstream inputFileStream(inputFileName, std::ios::binary);
  if (!inputFileStream.is_open()) {
    const std::string error =
        "audioFileToWordsFile() failed to open input file=" + inputFileName +
        " for reading";
    if (errorStream) {
      *errorStream << error << std::endl;
    } else {
      throw std::runtime_error(error);
    }
  }

  std::ofstream outputFileStream(outputFileName, std::ios::binary);
  if (!outputFileStream.is_open()) {
    const std::string error =
        "audioFileToWordsFile() failed to open output file=" + outputFileName +
        " for reading";
    if (errorStream) {
      *errorStream << error << std::endl;
    } else {
      throw std::runtime_error(error);
    }
  }

  return audioStreamToWordsStream(
      inputFileStream,
      outputFileStream,
      dnnModule,
      decoderFactory,
      decoderOptions,
      nTokens);
}

} // namespace

void audioFileToWordsFile(
    const std::string& inputFileName,
    const std::string& outputFileName,
    std::shared_ptr<streaming::Sequential> dnnModule,
    std::shared_ptr<const DecoderFactory> decoderFactory,
    const DecoderOptions& decoderOptions,
    int nTokens,
    std::ostream& errorStream) {
  audioFileToWordsFileImpl(
      inputFileName,
      outputFileName,
      dnnModule,
      decoderFactory,
      decoderOptions,
      nTokens,
      &errorStream);
}

void audioFileToWordsFile(
    const std::string& inputFileName,
    const std::string& outputFileName,
    std::shared_ptr<streaming::Sequential> dnnModule,
    std::shared_ptr<const DecoderFactory> decoderFactory,
    const DecoderOptions& decoderOptions,
    int nTokens) {
  audioFileToWordsFileImpl(
      inputFileName,
      outputFileName,
      dnnModule,
      decoderFactory,
      decoderOptions,
      nTokens,
      nullptr);
}

} // namespace streaming
} // namespace w2l
