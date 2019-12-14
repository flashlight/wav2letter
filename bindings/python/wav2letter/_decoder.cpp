/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libraries/decoder/WordLMDecoder.h"

#ifdef W2L_LIBRARIES_USE_KENLM
#include "libraries/lm/KenLM.h"
#endif

namespace py = pybind11;
using namespace w2l;
using namespace py::literals;

/**
 * Some hackery that lets pybind11 handle shared_ptr<void> (for LMStatePtr).
 * See: https://github.com/pybind/pybind11/issues/820
 */
PYBIND11_MAKE_OPAQUE(std::shared_ptr<void>);

namespace {

/**
 * A pybind11 "alias type" for abstract class LM, allowing one to subclass LM
 * with a custom LM defined purely in Python. For those who don't want to build
 * with KenLM, or have their own custom LM implementation.
 * See: https://pybind11.readthedocs.io/en/stable/advanced/classes.html
 *
 * Currently this works in principle, but is very slow due to
 * decoder calling `compareState` a huge number of times.
 *
 * TODO: ensure this works. Last time I tried this there were slicing issues,
 * see https://github.com/pybind/pybind11/issues/1546 for workarounds.
 * This is low-pri since we assume most people can just build with KenLM.
 */
class PyLM : public LM {
  using LM::LM;

  // needed for pybind11 or else it won't compile
  using LMOutput = std::pair<LMStatePtr, float>;

  LMStatePtr start(bool startWithNothing) override {
    PYBIND11_OVERLOAD_PURE(LMStatePtr, LM, start, startWithNothing);
  }

  LMOutput score(const LMStatePtr& state, const int usrTokenIdx) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, score, state, usrTokenIdx);
  }

  LMOutput finish(const LMStatePtr& state) override {
    PYBIND11_OVERLOAD_PURE(LMOutput, LM, finish, state);
  }

  int compareState(const LMStatePtr& state1, const LMStatePtr& state2)
      const override {
    PYBIND11_OVERLOAD_PURE(int, LM, compareState, state1, state2);
  }
};

void WordLMDecoder_decodeStep(
    WordLMDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  decoder.decodeStep(reinterpret_cast<const float*>(emissions), T, N);
}

std::vector<DecodeResult> WordLMDecoder_decode(
    WordLMDecoder& decoder,
    uintptr_t emissions,
    int T,
    int N) {
  return decoder.decode(reinterpret_cast<const float*>(emissions), T, N);
}

} // namespace

PYBIND11_MODULE(_decoder, m) {
  py::class_<std::shared_ptr<void>>(m, "encapsulated_data");

  py::enum_<SmearingMode>(m, "SmearingMode")
      .value("NONE", SmearingMode::NONE)
      .value("MAX", SmearingMode::MAX)
      .value("LOGADD", SmearingMode::LOGADD);

  py::class_<TrieNode, TrieNodePtr>(m, "TrieNode")
      .def(py::init<int>(), "idx"_a)
      .def_readwrite("children", &TrieNode::children)
      .def_readwrite("idx", &TrieNode::idx)
      .def_readwrite("labels", &TrieNode::labels)
      .def_readwrite("scores", &TrieNode::scores)
      .def_readwrite("max_score", &TrieNode::maxScore);

  py::class_<Trie, TriePtr>(m, "Trie")
      .def(py::init<int, int>(), "max_children"_a, "root_idx"_a)
      .def("get_root", &Trie::getRoot)
      .def("insert", &Trie::insert, "indices"_a, "label"_a, "score"_a)
      .def("search", &Trie::search, "indices"_a)
      .def("smear", &Trie::smear, "smear_mode"_a);

  py::class_<LM, LMPtr, PyLM>(m, "LM")
      .def(py::init<>())
      .def("start", &LM::start, "start_with_nothing"_a)
      .def("score", &LM::score, "state"_a, "usr_token_idx"_a)
      .def("finish", &LM::finish, "state"_a)
      .def("compare_state", &LM::compareState, "state1"_a, "state2"_a);

#ifdef W2L_LIBRARIES_USE_KENLM
  py::class_<KenLM, KenLMPtr, LM>(m, "KenLM")
      .def(
          py::init<const std::string&, const Dictionary&>(),
          "path"_a,
          "usr_token_dict"_a);
#endif

  py::enum_<CriterionType>(m, "CriterionType")
      .value("ASG", CriterionType::ASG)
      .value("CTC", CriterionType::CTC);

  py::class_<DecoderOptions>(m, "DecoderOptions")
      .def(
          py::init<
              const int,
              const int,
              const float,
              const float,
              const float,
              const float,
              const float,
              const float,
              const bool,
              const CriterionType>(),
          "beam_size"_a,
          "beam_size_token"_a,
          "beam_threshold"_a,
          "lm_weight"_a,
          "word_score"_a,
          "unk_score"_a,
          "sil_score"_a,
          "eos_score"_a,
          "log_add"_a,
          "criterion_type"_a)
      .def_readwrite("beam_size", &DecoderOptions::beamSize)
      .def_readwrite("beam_size_token", &DecoderOptions::beamSizeToken)
      .def_readwrite("beam_threshold", &DecoderOptions::beamThreshold)
      .def_readwrite("lm_weight", &DecoderOptions::lmWeight)
      .def_readwrite("word_score", &DecoderOptions::wordScore)
      .def_readwrite("unk_score", &DecoderOptions::unkScore)
      .def_readwrite("sil_score", &DecoderOptions::silScore)
      .def_readwrite("eos_score", &DecoderOptions::silScore)
      .def_readwrite("log_add", &DecoderOptions::logAdd)
      .def_readwrite("criterion_type", &DecoderOptions::criterionType);

  py::class_<DecodeResult>(m, "DecodeResult")
      .def(py::init<int>(), "length"_a)
      .def_readwrite("score", &DecodeResult::score)
      .def_readwrite("words", &DecodeResult::words)
      .def_readwrite("tokens", &DecodeResult::tokens);

  // NB: `decode` and `decodeStep` expect raw emissions pointers.
  py::class_<WordLMDecoder>(m, "WordLMDecoder")
      .def(py::init<
           const DecoderOptions&,
           const TriePtr,
           const LMPtr,
           const int,
           const int,
           const int,
           const std::vector<float>&>())
      .def("decode_begin", &WordLMDecoder::decodeBegin)
      .def(
          "decode_step", &WordLMDecoder_decodeStep, "emissions"_a, "T"_a, "N"_a)
      .def("decode_end", &WordLMDecoder::decodeEnd)
      .def("decode", &WordLMDecoder_decode, "emissions"_a, "T"_a, "N"_a)
      .def("prune", &WordLMDecoder::prune, "look_back"_a = 0)
      .def(
          "get_best_hypothesis",
          &WordLMDecoder::getBestHypothesis,
          "look_back"_a = 0)
      .def("get_all_final_hypothesis", &WordLMDecoder::getAllFinalHypothesis);
}
