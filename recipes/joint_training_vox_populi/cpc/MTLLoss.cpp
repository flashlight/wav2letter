// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#include <fstream>
#include <stdexcept>
#include <vector>

#include "flashlight/fl/autograd/Variable.h"
#include "flashlight/fl/common/Logging.h"
#include "flashlight/pkg/speech/runtime/runtime.h"

// MTL Loss flags
DEFINE_string(
    mtllossmapping,
    "",
    "Path to the MTL loss label mapping. Leave empty to not activate");

DEFINE_double(mtllossweight, 0.5, "Weight given to the MTL Loss");

typedef std::map<std::string, unsigned int> Mapping;

namespace asr4real {

Mapping loadMapping(const std::string& filename) {
  std::ifstream file(filename);
  unsigned int index = 0;
  Mapping output = Mapping();
  if (!file.is_open()) {
    throw std::invalid_argument("Cannot open " + filename);
  }
  std::string line;
  while (std::getline(file, line)) {
    output[line] = index;
    index++;
  }
  file.close();

  return output;
}

unsigned int getMapIndexFromFileID(
    const std::string& fileid,
    const Mapping& i_map) {
  size_t pos_delim = fileid.find('#');
  if (pos_delim == std::string::npos) {
    throw std::invalid_argument("Cannot parse " + fileid);
  }

  pos_delim++;
  const std::string token = fileid.substr(pos_delim, fileid.size() - pos_delim);
  return i_map.at(token);
}

af::array buildIndexLabels(
    const std::vector<af::array>& batch,
    const Mapping& i_map,
    const int batch_size) {
  af::array targets_lid_(1, batch_size, 1);
  for (int bIdx = 0; bIdx < batch_size; bIdx++) {
    auto filename =
        fl::pkg::speech::readSampleIds(batch[fl::pkg::speech::kSampleIdx])
            .at(bIdx);
    targets_lid_(0, bIdx, 0) = getMapIndexFromFileID(filename, i_map);
  }

  return targets_lid_;
}

fl::Variable mtl_step(
    fl::Variable& enc_out,
    std::shared_ptr<fl::Linear> crit,
    std::shared_ptr<fl::Dataset> trainset,
    const Mapping& i_map,
    const int batchIdx) {
  const int timedim = 1;
  const int batchdim = 2;
  const int featdim = 0;

  const int batchsz = enc_out.dims(batchdim);
  const int timesz = enc_out.dims(timedim);

  const std::vector<af::array>& batch = trainset->get(batchIdx);
  const fl::Variable target_ids_ =
      fl::Variable(buildIndexLabels(batch, i_map, batchsz), false);

  enc_out = fl::reorder(enc_out, featdim, timedim, batchdim);

  fl::Variable predictions = crit->forward(enc_out);
  predictions = fl::mean(predictions.as(f32), std::vector<int>{1}).as(f32);
  predictions = fl::logSoftmax(predictions, 0);
  fl::Variable loss = fl::categoricalCrossEntropy(
      predictions.as(f32), target_ids_, fl::ReduceMode::NONE);
  return fl::reorder(loss, 1, 0, 2);
}
}; // namespace asr4real
