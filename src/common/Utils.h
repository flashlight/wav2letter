/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Generic utilities which should not depend on ArrayFire / flashlight.
 */

#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/Defines.h"
#include "common/Dictionary.h"

namespace w2l {

// ============================ Types and Templates ============================

using LexiconMap =
    std::unordered_map<std::string, std::vector<std::vector<std::string>>>;

template <typename It>
using DecayDereference =
    typename std::decay<decltype(*std::declval<It>())>::type;

template <typename S, typename T>
using EnableIfSame = typename std::enable_if<std::is_same<S, T>::value>::type;

// ================================== Strings ==================================

std::string trim(const std::string& str);

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl);

bool startsWith(const std::string& input, const std::string& pattern);

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

/**
 * Join a vector of `std::string` inserting `delim` in between.
 */
std::string join(const std::string& delim, const std::vector<std::string>& vec);

/**
 * Join a range of `std::string` specified by iterators.
 */
template <
    typename FwdIt,
    typename = EnableIfSame<DecayDereference<FwdIt>, std::string>>
std::string join(const std::string& delim, FwdIt begin, FwdIt end);

/**
 * Create an output string using a `printf`-style format string and arguments.
 * Safer than `sprintf` which is vulnerable to buffer overflow.
 */
template <class... Args>
std::string format(const char* fmt, Args&&... args);

// ================================== System ==================================

std::string pathsConcat(const std::string& p1, const std::string& p2);

bool dirExists(const std::string& path);

void dirCreate(const std::string& path);

bool fileExists(const std::string& path);

std::string getEnvVar(const std::string& key, const std::string& dflt = "");

std::string getCurrentDate();

std::string getCurrentTime();

// =============================== Miscellaneous ===============================

std::vector<std::string> getFileContent(const std::string& file);

/**
 * Calls `f(args...)` repeatedly, retrying if an exception is thrown.
 * Supports sleeps between retries, with duration starting at `initial` and
 * multiplying by `factor` each retry. At most `maxIters` calls are made.
 */
template <class Fn, class... Args>
typename std::result_of<Fn(Args...)>::type retryWithBackoff(
    std::chrono::duration<double> initial,
    double factor,
    int64_t maxIters,
    Fn&& f,
    Args&&... args);

// ============================== Dataset helpers ==============================

// TODO: these should be moved to a more relevant location

std::vector<std::string> loadTarget(const std::string& filepath);

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

// ============================== Decoder helpers ==============================

// TODO: these should be moved to a more relevant location

Dictionary createWordDict(const LexiconMap& lexicon);

LexiconMap loadWords(const std::string& fn, const int64_t maxNumWords);

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

#include "Utils-inl.h"
