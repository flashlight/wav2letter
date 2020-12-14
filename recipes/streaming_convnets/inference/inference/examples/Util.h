/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cereal/archives/json.hpp>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "flashlight/lib/text/decoder/Decoder.h"
#include "inference/common/IOBuffer.h"

namespace w2l {
namespace streaming {

// Dumps a message on start and finish (on destruction) with formatted elapsed
// time.
class TimeElapsedReporter {
 public:
  explicit TimeElapsedReporter(std::string name);
  ~TimeElapsedReporter();

 private:
  const std::string name_;
  const std::chrono::time_point<std::chrono::high_resolution_clock>
      startTimepoint_;
};

// If fileName is a full path then return it as is. Othereise, prefix the
// fileName with path and adds a separator between if needed.
std::string GetFullPath(const std::string& fileName, const std::string& path);

// Returns the filename part including its extension.
std::string getFileName(const std::string& path);

// Retutns number of bytes read.
int readStreamIntoBuffer(
    std::istream& inputStream,
    std::shared_ptr<IOBuffer> buffer,
    int bytesToRead);

// Read type StreamType off inputStream, apply transformationFunction to each
// element and write the tranformed elemnt into buffer. Retutns the number os
// BufferType elements written into buffer.
template <typename StreamType, typename BufferType>
int readTransformStreamIntoBuffer(
    std::istream& inputStream,
    std::shared_ptr<IOBuffer> buffer,
    int sizeInBufferType,
    const std::function<BufferType(StreamType)>& transformationFunction) {
  const int sizeInBytes = sizeInBufferType * sizeof(BufferType);
  auto tmpBuffer = std::make_shared<IOBuffer>(sizeInBytes);
  const int bytesRead =
      readStreamIntoBuffer(inputStream, tmpBuffer, sizeInBytes);

  int16_t* tmpPtr = tmpBuffer->data<int16_t>();
  const int tmpSize = tmpBuffer->size<int16_t>();

  buffer->ensure<float>(tmpSize);
  float* bufferPtr = buffer->data<float>();
  std::transform(tmpPtr, tmpPtr + tmpSize, bufferPtr, transformationFunction);
  if (bytesRead % sizeof(StreamType)) {
    std::cerr << "readTransformIntoBuffer(buffer=" << buffer->debugString()
              << " ,sizeInBufferType=" << sizeInBufferType << ") read "
              << bytesRead << " bytes that is not devisible by "
              << sizeof(StreamType) << std::endl;
  }
  buffer->move<float>(tmpSize);
  return tmpSize;
}

} // namespace streaming
} // namespace w2l

namespace cereal {

template <typename Archive>
inline std::string save_minimal(
    const Archive&,
    const fl::lib::text::CriterionType& criterionType) {
  switch (criterionType) {
    case fl::lib::text::CriterionType::ASG:
      return "ASG";
    case fl::lib::text::CriterionType::CTC:
      return "CTC";
    case fl::lib::text::CriterionType::S2S:
      return "S2S";
  }
  throw std::runtime_error(
      "save_minimal() got invalid CriterionType value=" +
      std::to_string(static_cast<int>(criterionType)));
}

template <typename Archive>
void load_minimal(
    const Archive&,
    fl::lib::text::CriterionType& obj,
    const std::string& value) {
  if (value == "ASG") {
    obj = fl::lib::text::CriterionType::ASG;
  } else if (value == "CTC") {
    obj = fl::lib::text::CriterionType::CTC;
  } else if (value == "S2S") {
    obj = fl::lib::text::CriterionType::S2S;
  } else {
    throw std::runtime_error(
        "load_minimal() got invalid CriterionType value=" + value);
  }
}
} // namespace cereal
