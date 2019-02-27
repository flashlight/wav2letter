/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>

#include <flashlight/flashlight.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Utils.h"

namespace w2l {

struct W2lSerializer {
 public:
  template <class... Args>
  static void save(const std::string& filepath, const Args&... args) {
    retryWithBackoff(
        std::chrono::seconds(1),
        2.0,
        6,
        saveImpl<Args...>,
        filepath,
        args...); // max wait 31s
  }

  template <typename... Args>
  static void load(const std::string& filepath, Args&... args) {
    retryWithBackoff(
        std::chrono::seconds(1),
        2.0,
        6,
        loadImpl<Args...>,
        filepath,
        args...); // max wait 31s
  }

 private:
  template <typename... Args>
  static void saveImpl(const std::string& filepath, const Args&... args) {
    try {
      std::ofstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error("failed to open file for writing");
      }
      cereal::BinaryOutputArchive ar(file);
      ar(std::string(W2L_VERSION));
      ar(args...);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error while saving: " << ex.what() << "\n";
      throw;
    }
  }

  template <typename... Args>
  static void loadImpl(const std::string& filepath, Args&... args) {
    try {
      std::ifstream file(filepath, std::ios::binary);
      if (!file.is_open()) {
        throw std::runtime_error("failed to open file for reading");
      }
      std::string version;
      cereal::BinaryInputArchive ar(file);
      ar(version);
      ar(args...);
    } catch (const std::exception& ex) {
      LOG(ERROR) << "Error while loading: " << ex.what() << "\n";
      throw;
    }
  }
};

std::string newRunPath(
    const std::string& root,
    const std::string& runname = "",
    const std::string& tag = "");

std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath);

/**
 * Given a filename, remove any filepath delimiters - returns a contiguous
 * string that won't be subdivided into a filepath
 */
std::string cleanFilepath(const std::string& in);

} // namespace w2l
