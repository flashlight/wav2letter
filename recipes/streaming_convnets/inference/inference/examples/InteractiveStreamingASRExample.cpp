/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Summary
 * --------
 * Interactive tiny shell for quickly transcribing audio files on the fly.
 *
 * User guide
 * ----------
 *
 * 1. Setup the input files:
 * Assuming that you have the acoustic model, language model, features
 * extraction serialized streaming inference DNN, tokens file, lexicon file and
 * input audio file in a directory called model.
 *
 *  $> ls ~/model
 *   acoustic_model.bin
 *   language.bin
 *   feat.bin
 *   tokens.txt
 *   lexicon.txt
 *
 * $> ls ~/audio
 *   input1.wav
 *
 * 2. Run
 *
 * .$> interactive_streaming_asr_example --input_files_base_path ~/model/
 * Started features model file loading ...
 * Completed features model file loading elapsed time=46557 microseconds
 *
 * Started acoustic model file loading ...
 * Completed acoustic model file loading elapsed time=2058 milliseconds
 *
 * Started tokens file loading ...
 * Completed tokens file loading elapsed time=1318 microseconds
 *
 * Tokens loaded - 9998 tokens
 * Started decoder options file loading ...
 * Completed decoder options file loading elapsed time=388 microseconds
 *
 * Started create decoder ...
 * [Letters] 9998 tokens loaded.
 * [Words] 200001 words loaded.
 * Completed create decoder elapsed time=884 milliseconds
 *
 * Entering interactive command line shell. enter '?' for help.
 * ------------------------------------------------------------
 * $>input=/home/audio.wav
 * #start (msec), end(msec), transcription
 * 0,1000,
 * 1000,2000,i wish he
 * 2000,3000,had never been to school
 * 3000,4000,missus
 * 4000,4260,began again brusquely
 * Completed create decoder elapsed time=2760 milliseconds
 *
 */

#include <fstream>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <gflags/gflags.h>

#include "inference/decoder/Decoder.h"
#include "inference/examples/AudioToWords.h"
#include "inference/examples/Util.h"
#include "inference/module/feature/feature.h"
#include "inference/module/module.h"
#include "inference/module/nn/nn.h"

using namespace w2l;
using namespace w2l::streaming;

DEFINE_string(
    input_files_base_path,
    ".",
    "path is added as prefix to input files unless the input file"
    " is a full path.");
DEFINE_string(
    feature_module_file,
    "feature_extractor.bin",
    "serialized feature extraction module.");
DEFINE_string(
    acoustic_module_file,
    "acoustic_model.bin",
    "binary file containing acoustic module parameters.");
DEFINE_string(
    transitions_file,
    "",
    "binary file containing ASG criterion transition parameters.");
DEFINE_string(tokens_file, "tokens.txt", "text file containing tokens.");
DEFINE_string(lexicon_file, "lexicon.txt", "text file containing lexicon.");
DEFINE_string(silence_token, "_", "the token to use to denote silence");
DEFINE_string(
    language_model_file,
    "language_model.bin",
    "binary file containing language module parameters.");
DEFINE_string(
    decoder_options_file,
    "decoder_options.json",
    "JSON file containing decoder options"
    " including: max overall beam size, max beam for token selection, beam score threshold"
    ", language model weight, word insertion score, unknown word insertion score"
    ", silence insertion score, and use logadd when merging decoder nodes");

std::string GetInputFileFullPath(const std::string& fileName) {
  return GetFullPath(fileName, FLAGS_input_files_base_path);
}

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  std::shared_ptr<streaming::Sequential> featureModule;
  std::shared_ptr<streaming::Sequential> acousticModule;

  // Read files
  {
    TimeElapsedReporter feturesLoadingElapsed("features model file loading");
    std::ifstream featFile(
        GetInputFileFullPath(FLAGS_feature_module_file), std::ios::binary);
    if (!featFile.is_open()) {
      throw std::runtime_error(
          "failed to open feature file=" +
          GetInputFileFullPath(FLAGS_feature_module_file) + " for reading");
    }
    cereal::BinaryInputArchive ar(featFile);
    ar(featureModule);
  }

  {
    TimeElapsedReporter acousticLoadingElapsed("acoustic model file loading");
    std::ifstream amFile(
        GetInputFileFullPath(FLAGS_acoustic_module_file), std::ios::binary);
    if (!amFile.is_open()) {
      throw std::runtime_error(
          "failed to open acoustic model file=" +
          GetInputFileFullPath(FLAGS_feature_module_file) + " for reading");
    }
    cereal::BinaryInputArchive ar(amFile);
    ar(acousticModule);
  }

  // String both modeles togthers to a single DNN.
  auto dnnModule = std::make_shared<streaming::Sequential>();
  dnnModule->add(featureModule);
  dnnModule->add(acousticModule);

  std::vector<std::string> tokens;
  {
    TimeElapsedReporter acousticLoadingElapsed("tokens file loading");
    std::ifstream tknFile(GetInputFileFullPath(FLAGS_tokens_file));
    if (!tknFile.is_open()) {
      throw std::runtime_error(
          "failed to open tokens file=" +
          GetInputFileFullPath(FLAGS_tokens_file) + " for reading");
    }
    std::string line;
    while (std::getline(tknFile, line)) {
      tokens.push_back(line);
    }
  }
  int nTokens = tokens.size();
  std::cout << "Tokens loaded - " << nTokens << " tokens" << std::endl;

  fl::lib::text::LexiconDecoderOptions decoderOptions;
  {
    TimeElapsedReporter decoderOptionsElapsed("decoder options file loading");
    std::ifstream decoderOptionsFile(
        GetInputFileFullPath(FLAGS_decoder_options_file));
    if (!decoderOptionsFile.is_open()) {
      throw std::runtime_error(
          "failed to open decoder options file=" +
          GetInputFileFullPath(FLAGS_decoder_options_file) + " for reading");
    }
    cereal::JSONInputArchive ar(decoderOptionsFile);
    // TODO: factor out proper serialization functionality or Cereal
    // specialization.
    ar(cereal::make_nvp("beamSize", decoderOptions.beamSize),
       cereal::make_nvp("beamSizeToken", decoderOptions.beamSizeToken),
       cereal::make_nvp("beamThreshold", decoderOptions.beamThreshold),
       cereal::make_nvp("lmWeight", decoderOptions.lmWeight),
       cereal::make_nvp("wordScore", decoderOptions.wordScore),
       cereal::make_nvp("unkScore", decoderOptions.unkScore),
       cereal::make_nvp("silScore", decoderOptions.silScore),
       cereal::make_nvp("logAdd", decoderOptions.logAdd),
       cereal::make_nvp("criterionType", decoderOptions.criterionType));
  }

  std::vector<float> transitions;
  if (!FLAGS_transitions_file.empty()) {
    TimeElapsedReporter acousticLoadingElapsed("transitions file loading");
    std::ifstream transitionsFile(
        GetInputFileFullPath(FLAGS_transitions_file), std::ios::binary);
    if (!transitionsFile.is_open()) {
      throw std::runtime_error(
          "failed to open transition parameter file=" +
          GetInputFileFullPath(FLAGS_transitions_file) + " for reading");
    }
    cereal::BinaryInputArchive ar(transitionsFile);
    ar(transitions);
  }

  std::shared_ptr<const DecoderFactory> decoderFactory;
  // Create Decoder
  {
    TimeElapsedReporter acousticLoadingElapsed("create decoder");
    decoderFactory = std::make_shared<DecoderFactory>(
        GetInputFileFullPath(FLAGS_tokens_file),
        GetInputFileFullPath(FLAGS_lexicon_file),
        GetInputFileFullPath(FLAGS_language_model_file),
        transitions,
        fl::lib::text::SmearingMode::MAX,
        FLAGS_silence_token,
        0);
  }

  const std::string inputFilecommand = "input=";
  const std::string outputFilecommand = "output=";
  const std::string setEndTokencommand = "endtoken=";
  std::string inputFilename;
  std::string outputFilename = "stdout";
  std::ostream* outStream = &std::cout;
  std::ofstream outputFileStream;
  std::string endToken = "#finish transcribing";
  std::cout << "Entering interactive command line shell. enter '?' for help.\n";
  std::cout << "------------------------------------------------------------\n";
  while (true) {
    std::string cmdline;
    std::cout << "$>";
    std::getline(std::cin, cmdline);
    if (cmdline == "?" || cmdline == "help") {
      std::cout
          << "Interactive streaming ASR shell:\n"
          << "-----------------------------------------------------------\n"
          << "? or help         to print this message.\n"
          << "input=[filename]  transcribe the given audio file.\n"
          << "output=[filename] write transcription to output file.\n"
          << "output=stdout     write transcription to stdout.\n"
          << "endtoken=[token]  set string that marks end of transciption.\n"
          << "exit or q         exit this shell.\n";
    } else if (cmdline == "exit" || cmdline == "q") {
      break;
    } else if (cmdline.rfind(setEndTokencommand, 0) == 0) {
      endToken = cmdline.substr(setEndTokencommand.size());
      std::cout << "End of trascription token=" << endToken << std::endl;
    } else if (cmdline.rfind(outputFilecommand, 0) == 0) {
      outputFilename = cmdline.substr(outputFilecommand.size());
      if (outputFilename == "stdout") {
        outStream = &std::cout;
      } else {
        outputFileStream.close();
        outputFileStream.open(
            outputFilename, std::ofstream::out | std::ofstream::app);
        if (outputFileStream.good()) {
          outStream = &outputFileStream;
        } else {
          std::cerr << "Failed to open file:" << outputFilename
                    << " for writing. Defaulting to stdout.\n";
          outStream = &std::cout;
        }
      }
      std::cout << "Redirecting trascription output to:" << outputFilename
                << std::endl;
    } else if (cmdline.rfind(inputFilecommand, 0) == 0) {
      inputFilename = cmdline.substr(inputFilecommand.size());
      std::ifstream audioFile(inputFilename, std::ios::binary);
      *outStream << "Transcribing file:" << inputFilename
                 << " to:" << outputFilename << std::endl;
      audioStreamToWordsStream(
          audioFile,
          *outStream,
          dnnModule,
          decoderFactory,
          decoderOptions,
          nTokens);
      *outStream << endToken << std::endl;
    } else {
      std::cout << "unknown command:" << cmdline << std::endl;
    }
  }
}
