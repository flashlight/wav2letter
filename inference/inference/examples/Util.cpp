/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 */

#include "inference/examples/Util.h"

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

namespace w2l {
namespace streaming {

#ifdef _WIN32
constexpr const char* separator = "\\";
#else
constexpr const char* separator = "/";
#endif

std::string prettyDuration(
    const std::chrono::time_point<std::chrono::high_resolution_clock>& start,
    const std::chrono::time_point<std::chrono::high_resolution_clock>& end) {
  const auto runtime = end - start;
  auto runtimeMicroSec =
      std::chrono::duration_cast<std::chrono::microseconds>(runtime);
  auto runtimeMiliSec =
      std::chrono::duration_cast<std::chrono::milliseconds>(runtime);
  auto runtimeSeconds =
      std::chrono::duration_cast<std::chrono::seconds>(runtime);
  std::stringstream strStream;
  strStream << " elapsed time=";
  if (runtimeMicroSec.count() < 1e5) {
    strStream << runtimeMicroSec.count() << " microseconds\n";
  } else if (runtimeMiliSec.count() < 1e5) {
    strStream << runtimeMiliSec.count() << " milliseconds\n";
  } else {
    strStream << runtimeSeconds.count() << " seconds\n";
  }
  return strStream.str();
}

TimeElapsedReporter::TimeElapsedReporter(std::string name)
    : name_(std::move(name)),
      startTimepoint_(std::chrono::high_resolution_clock::now()) {
  std::cout << "Started " << name_ << " ... " << std::endl;
}

TimeElapsedReporter::~TimeElapsedReporter() {
  std::cout << "Completed " << name_
            << prettyDuration(
                   startTimepoint_, std::chrono::high_resolution_clock::now())
            << std::endl;
}

std::string GetFullPath(const std::string& fileName, const std::string& path) {
  // If fileName is a full path then return it as is.
  if (!fileName.empty() && fileName[0] == separator[0]) {
    return fileName;
  }
  const std::string requiredSeperator =
      (*path.rbegin() == separator[0]) ? "" : separator;

  return path + requiredSeperator + fileName;
}

std::string getFileName(const std::string& path) {
  const size_t separatorIndex = path.rfind(separator, path.length());
  if (separatorIndex == std::string::npos) {
    return path;
  }
  return path.substr(separatorIndex + 1, path.length() - separatorIndex);
}

int readStreamIntoBuffer(
    std::istream& inputStream,
    std::shared_ptr<IOBuffer> buffer,
    int bytesToRead) {
  assert(bytesToRead > 0);
  assert(buffer);
  int bytesRead = 0;
  buffer->ensure<char>(bytesToRead);
  char* inputPtrChar = buffer->data<char>();
  while (bytesRead < bytesToRead && inputStream.good()) {
    inputStream.read(inputPtrChar + bytesRead, bytesToRead - bytesRead);
    bytesRead += inputStream.gcount();
  }
  buffer->move<char>(bytesRead);
  return bytesRead;
}

} // namespace streaming
} // namespace w2l
