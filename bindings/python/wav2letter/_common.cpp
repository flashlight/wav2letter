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
      .def(py::init<const std::string&>(), "filename"_a)
      .def("entrySize", &Dictionary::entrySize)
      .def("indexSize", &Dictionary::indexSize)
      .def("addEntry", &Dictionary_addEntry_0, "entry"_a, "idx"_a)
      .def("addEntry", &Dictionary_addEntry_1, "entry"_a)
      .def("getEntry", &Dictionary::getEntry, "idx"_a)
      .def("setDefaultIndex", &Dictionary::setDefaultIndex, "idx"_a)
      .def("getIndex", &Dictionary::getIndex, "entry"_a)
      .def("contains", &Dictionary::contains, "entry"_a)
      .def("isContiguous", &Dictionary::isContiguous)
      .def("mapEntriesToIndices", &Dictionary::mapEntriesToIndices, "entries"_a)
      .def(
          "mapIndicesToEntries", &Dictionary::mapIndicesToEntries, "indices"_a);

  m.def("createWordDict", &createWordDict, "lexicon"_a);
  m.def("loadWords", &loadWords, "filename"_a, "maxWords"_a = -1);
  m.def("tkn2Idx", &tkn2Idx, "spelling"_a, "tokenDict"_a, "maxReps"_a);
  m.def("packReplabels", &packReplabels, "tokens"_a, "dict"_a, "maxReps"_a);
  m.def("unpackReplabels", &unpackReplabels, "tokens"_a, "dict"_a, "maxReps"_a);
}
