/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "Serial.h"

#include "common/FlashlightUtils.h"

#include <fstream>
#include <sstream>

namespace w2l {

std::string newRunPath(
    const std::string& root,
    const std::string& runname /* = "" */,
    const std::string& tag /* = "" */) {
  std::string dir = "";
  if (runname.empty()) {
    auto dt = getCurrentDate();
    std::string tm = getCurrentTime();
    replaceAll(tm, ":", "-");
    dir += (dt + "_" + tm + "_" + getEnvVar("HOSTNAME", "unknown_host") + "_");

    // Unique hash based on config
    auto hash = std::hash<std::string>{}(serializeGflags());
    dir += std::to_string(hash);

  } else {
    dir += runname;
  }
  if (!tag.empty()) {
    dir += "_" + tag;
  }
  return pathsConcat(root, dir);
}

std::string
getRunFile(const std::string& name, int runidx, const std::string& runpath) {
  auto fname = format("%03d_%s", runidx, name.c_str());
  return pathsConcat(runpath, fname);
};

std::string cleanFilepath(const std::string& in) {
  std::string replace = in;
  std::string sep = "/";
#ifdef _WIN32
  sep = "\\";
#endif
  replaceAll(replace, sep, "#");
  return replace;
}

std::string serializeGflags(const std::string& separator /* = "\n" */) {
  std::stringstream serialized;
  std::vector<gflags::CommandLineFlagInfo> allFlags;
  gflags::GetAllFlags(&allFlags);
  std::string currVal;
  auto& deprecatedFlags = detail::getDeprecatedFlags();
  for (auto itr = allFlags.begin(); itr != allFlags.end(); ++itr) {
    // Check if the flag is deprecated - if so, skip it
    if (deprecatedFlags.find(itr->name) == deprecatedFlags.end()) {
      gflags::GetCommandLineOption(itr->name.c_str(), &currVal);
      serialized << "--" << itr->name << "=" << currVal << separator;
    }
  }
  return serialized.str();
}

} // namespace w2l
