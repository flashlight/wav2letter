/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "libraries/common/Dictionary.h"
#include "libraries/common/WordUtils.h"

namespace py = pybind11;
using namespace w2l;
using namespace py::literals;

namespace {

void Dictionary_addEntry_0(
    Dictionary& dict,
    const std::string& entry,
    int idx) {
  dict.addEntry(entry, idx);
}

void Dictionary_addEntry_1(Dictionary& dict, const std::string& entry) {
  dict.addEntry(entry);
}

} // namespace

PYBIND11_MODULE(_common, m) {
  py::class_<Dictionary>(m, "Dictionary")
      .def(py::init<>())
      .def(py::init<const std::string&>(), "filename"_a)
      .def("entry_size", &Dictionary::entrySize)
      .def("index_size", &Dictionary::indexSize)
      .def("add_entry", &Dictionary_addEntry_0, "entry"_a, "idx"_a)
      .def("add_entry", &Dictionary_addEntry_1, "entry"_a)
      .def("get_entry", &Dictionary::getEntry, "idx"_a)
      .def("set_default_index", &Dictionary::setDefaultIndex, "idx"_a)
      .def("get_index", &Dictionary::getIndex, "entry"_a)
      .def("contains", &Dictionary::contains, "entry"_a)
      .def("is_contiguous", &Dictionary::isContiguous)
      .def(
          "map_entries_to_indices",
          &Dictionary::mapEntriesToIndices,
          "entries"_a)
      .def(
          "map_indices_to_entries",
          &Dictionary::mapIndicesToEntries,
          "indices"_a);

  m.def("create_word_dict", &createWordDict, "lexicon"_a);
  m.def("load_words", &loadWords, "filename"_a, "max_words"_a = -1);
  m.def("tkn_to_idx", &tkn2Idx, "spelling"_a, "token_dict"_a, "max_reps"_a);
  m.def("pack_replabels", &packReplabels, "tokens"_a, "dict"_a, "max_reps"_a);
  m.def(
      "unpack_replabels", &unpackReplabels, "tokens"_a, "dict"_a, "max_reps"_a);
}
