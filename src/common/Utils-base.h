/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <arrayfire.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/Dictionary.h"

namespace w2l {

typedef std::unordered_map<std::string, std::vector<std::vector<std::string>>>
    LexiconMap;

std::string pathsConcat(const std::string& p1, const std::string& p2);

std::string trim(const std::string& str);

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl);

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty = false);

std::vector<std::string> split(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty = false);

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty = false);

bool dirExists(const std::string& path);

void dirCreate(const std::string& path);

bool fileExists(const std::string& path);

std::string getEnvVar(const std::string& key, const std::string& dflt = "");

std::string getCurrentDate();

std::string getCurrentTime();

std::string serializeGflags(const std::string& separator = "\n");

std::vector<std::string> getFileContent(const std::string& file);

bool startsWith(const std::string& input, const std::string& pattern);

std::vector<std::string> loadTarget(const std::string& filepath);

int64_t loadSize(const std::string& filepath);

Dictionary createTokenDict(const std::string& filepath);
Dictionary createTokenDict();

Dictionary createWordDict(const LexiconMap& lexicon);

std::vector<std::string> wrd2Target(
    const std::string& word,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

std::vector<std::string> wrd2Target(
    const std::vector<std::string>& words,
    const LexiconMap& lexicon,
    const Dictionary& dict,
    bool fallback2Ltr = false,
    bool skipUnk = false);

/************** Decoder helpers **************/
LexiconMap loadWords(const std::string& fn, const int64_t maxNumWords);

Dictionary createLMDict(const std::string& filepath);

// split word into tokens abc -> {"a", "b", "c"}
// Works with ASCII, UTF-8 encodings
std::vector<std::string> splitWrd(const std::string& word);

/* A series of vector to vector mapping operations */
std::vector<int> tkn2Idx(const std::vector<std::string>&, const Dictionary&);

std::vector<int> validateIdx(std::vector<int>, const int);

std::vector<std::string> tknIdx2Ltr(const std::vector<int>&, const Dictionary&);

std::vector<std::string> tkn2Wrd(const std::vector<std::string>&);

// will be deprecated soon
std::vector<std::string> wrdIdx2Wrd(const std::vector<int>&, const Dictionary&);

std::vector<std::string> tknTarget2Ltr(std::vector<int>, const Dictionary&);

std::vector<std::string> tknPrediction2Ltr(std::vector<int>, const Dictionary&);

} // namespace w2l
