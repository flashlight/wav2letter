/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <thread>
#include <unordered_map>

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <flashlight/flashlight.h>
#include <glog/logging.h>

#include "common/Defines.h"
#include "common/Utils.h"

namespace w2l {

class W2lSerializer {
 public:
  template <typename... Args>
  static void save(const std::string& filepath, Args&&... args) {
    auto attemptsLeft = kTryCatchAttempts;
    do {
      try {
        std::ofstream file(filepath, std::ios::binary);
        file.clear();
        {
          cereal::BinaryOutputArchive ar(file);
          ar(std::string(W2L_VERSION));
          ar(std::forward<Args>(args)...);
        }
        file.close();
        break;
      } catch (const std::exception& ex) {
        --attemptsLeft;
        LOG_IF(FATAL, attemptsLeft <= 0)
            << "Error while saving to file '" << filepath << "': " << ex.what();
        std::this_thread::sleep_for(std::chrono::seconds(kTryCatchWaitSec));
      }
    } while (attemptsLeft > 0);
  }

  template <typename... Args>
  static void load(const std::string& filepath, Args&... args) {
    LOG_IF(FATAL, !fileExists(filepath))
        << "File - '" << filepath << "' does not exist.";
    auto attemptsLeft = kTryCatchAttempts;
    do {
      try {
        std::ifstream file(filepath, std::ios::binary);
        std::string version;
        {
          cereal::BinaryInputArchive ar(file);
          ar(version);
          ar(args...);
        }
        file.close();
        break;
      } catch (const std::exception& ex) {
        --attemptsLeft;
        LOG_IF(FATAL, attemptsLeft <= 0)
            << "Error while loading file - '" << filepath << "': " << ex.what();
        std::this_thread::sleep_for(std::chrono::seconds(kTryCatchWaitSec));
      }
    } while (attemptsLeft > 0);
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
