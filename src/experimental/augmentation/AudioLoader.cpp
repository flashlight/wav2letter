// Copyright 2004-present Facebook. All Rights Reserved.

#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "data/Sound.h"
#include "experimental/augmentation/AudioLoader.h"
#include "flashlight/common/CppBackports.h"

namespace w2l {
namespace augmentation {

#ifdef _WIN32
constexpr const char* separator = "\\";
#else
constexpr const char* separator = "/";
#endif

std::string GetFullPath(const std::string& path, const std::string& fileName) {
  // If fileName is a full path then return it as is.
  if (!fileName.empty() && fileName[0] == separator[0]) {
    return fileName;
  }
  const std::string requiredSeperator =
      (*path.rbegin() == separator[0]) ? "" : separator;

  return path + requiredSeperator + fileName;
}

AudioLoader::AudioLoader(const std::string& audioDirectoryPath)
    : audioDirectoryPath_(audioDirectoryPath) {
  std::stringstream ss;
  ss << "AudioLoader::AudioLoader(audioDirectoryPath=" << audioDirectoryPath
     << ")";
  DIR* dir = opendir(audioDirectoryPath.c_str());
  if (!dir) {
    std::stringstream ss;
    ss << " failed to open directory";
    throw std::invalid_argument(ss.str());
  }
  for (struct dirent* entry = readdir(dir); entry; entry = readdir(dir)) {
    std::string filename = entry->d_name;
    if (filename[0] != '.') {
      audioFilePathVec_.push_back(std::move(filename));
    }
  }
  closedir(dir);

  ss << " found " << audioFilePathVec_.size() << " audio files.";
  std::cout << ss.str() << std::endl;

  uniformDistribution_ = fl::cpp::make_unique<std::uniform_int_distribution<>>(
      0, audioFilePathVec_.size() - 1);
}

AudioLoader::Audio AudioLoader::loadRandom() {
  const int randomIndex = (*uniformDistribution_)(randomEngine_);
  AudioLoader::Audio result;
  result.filename_ = audioFilePathVec_[randomIndex];
  result.fullpath_ = GetFullPath(audioDirectoryPath_, result.filename_);
  try {
    result.data_ = w2l::loadSound<float>(result.fullpath_);
    result.info_ = w2l::loadSoundInfo(result.fullpath_);
  } catch (std::exception& ex) {
    std::stringstream ss;
    ss << "AudioLoader::getRandomAudioTrack() failed to load audio file="
       << result.fullpath_ << " with error=" << ex.what();
    throw std::runtime_error(ss.str());
  }

  return result;
}

std::string AudioLoader::Audio::prettyString() const {
  std::stringstream ss;
  ss << "data_.size()=" << data_.size() << " info_={}"
     << " filename_=" << filename_ << " fullpath_=" << fullpath_;
  return ss.str();
}

} // namespace augmentation
} // namespace w2l
