/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "libraries/common/Utils.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <array>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <functional>

static constexpr const char* kSpaceChars = "\t\n\v\f\r ";

namespace w2l {

std::string trim(const std::string& str) {
  auto i = str.find_first_not_of(kSpaceChars);
  if (i == std::string::npos) {
    return "";
  }
  auto j = str.find_last_not_of(kSpaceChars);
  if (j == std::string::npos || i > j) {
    return "";
  }
  return str.substr(i, j - i + 1);
}

void replaceAll(
    std::string& str,
    const std::string& from,
    const std::string& repl) {
  if (from.empty()) {
    return;
  }
  size_t pos = 0;
  while ((pos = str.find(from, pos)) != std::string::npos) {
    str.replace(pos, from.length(), repl);
    pos += repl.length();
  }
}

bool startsWith(const std::string& input, const std::string& pattern) {
  return (input.find(pattern) == 0);
}

template <bool Any, typename Delim>
static std::vector<std::string> splitImpl(
    const Delim& delim,
    std::string::size_type delimSize,
    const std::string& input,
    bool ignoreEmpty = false) {
  std::vector<std::string> result;
  std::string::size_type i = 0;
  while (true) {
    auto j = Any ? input.find_first_of(delim, i) : input.find(delim, i);
    if (j == std::string::npos) {
      break;
    }
    if (!(ignoreEmpty && i == j)) {
      result.emplace_back(input.begin() + i, input.begin() + j);
    }
    i = j + delimSize;
  }
  if (!(ignoreEmpty && i == input.size())) {
    result.emplace_back(input.begin() + i, input.end());
  }
  return result;
}

std::vector<std::string>
split(char delim, const std::string& input, bool ignoreEmpty) {
  return splitImpl<false>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string>
split(const std::string& delim, const std::string& input, bool ignoreEmpty) {
  if (delim.empty()) {
    throw std::invalid_argument("delimiter is empty string");
  }
  return splitImpl<false>(delim, delim.size(), input, ignoreEmpty);
}

std::vector<std::string> splitOnAnyOf(
    const std::string& delim,
    const std::string& input,
    bool ignoreEmpty) {
  return splitImpl<true>(delim, 1, input, ignoreEmpty);
}

std::vector<std::string> splitOnWhitespace(
    const std::string& input,
    bool ignoreEmpty) {
  return splitOnAnyOf(kSpaceChars, input, ignoreEmpty);
}

std::string join(
    const std::string& delim,
    const std::vector<std::string>& vec) {
  return join(delim, vec.begin(), vec.end());
}

std::string pathsConcat(const std::string& p1, const std::string& p2) {
  char sep = '/';

#ifdef _WIN32
  sep = '\\';
#endif

  if (!p1.empty() && p1[p1.length() - 1] != sep) {
    return (trim(p1) + sep + trim(p2)); // Need to add a path separator
  } else {
    return (trim(p1) + trim(p2));
  }
}

bool dirExists(const std::string& path) {
  struct stat info;
  if (stat(path.c_str(), &info) != 0) {
    return false;
  } else if (info.st_mode & S_IFDIR) {
    return true;
  } else {
    return false;
  }
}

void dirCreate(const std::string& path) {
  if (dirExists(path)) {
    return;
  }
  mode_t nMode = 0755;
  int nError = 0;
#ifdef _WIN32
  nError = _mkdir(path.c_str());
#else
  nError = mkdir(path.c_str(), nMode);
#endif
  if (nError != 0) {
    throw std::runtime_error(
        std::string() + "Unable to create directory - " + path);
  }
}

bool fileExists(const std::string& path) {
  std::ifstream fs(path, std::ifstream::in);
  return fs.good();
}

std::string getEnvVar(
    const std::string& key,
    const std::string& dflt /*= "" */) {
  char* val = getenv(key.c_str());
  return val ? std::string(val) : dflt;
}

std::string getCurrentDate() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%Y-%m-%d", tstruct);
  return std::string(buf.data());
}

std::string getCurrentTime() {
  time_t now = time(nullptr);
  struct tm tmbuf;
  struct tm* tstruct;
  tstruct = localtime_r(&now, &tmbuf);

  std::array<char, 80> buf;
  strftime(buf.data(), buf.size(), "%X", tstruct);
  return std::string(buf.data());
}

std::vector<std::string> getFileContent(const std::string& file) {
  std::vector<std::string> data;
  std::ifstream in(file);
  if (!in.good()) {
    throw std::runtime_error(
        std::string() + "Could not read file '" + file + "'");
  }
  std::string str;
  while (std::getline(in, str)) {
    data.emplace_back(str);
  }
  in.close();
  return data;
}

} // namespace w2l
