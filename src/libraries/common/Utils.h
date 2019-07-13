/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <chrono>
#include <cstring>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace w2l {

// ============================ Types and Templates ============================

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

/// Zeroes `count * sizeof(T)` bytes
template <typename T>
void setZero(T* ptr, size_t count) {
  std::memset(ptr, 0, count * sizeof(T));
}

} // namespace w2l

#include "libraries/common/Utils-inl.h"
