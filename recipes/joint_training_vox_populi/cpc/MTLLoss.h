// (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

#pragma once
#include <gflags/gflags.h>
#include <map>
#include <string>
#include <vector>
#include "flashlight/fl/contrib/modules/modules.h"
#include "flashlight/fl/nn/modules/Linear.h"
#include "flashlight/pkg/speech/common/Defines.h"
#include "flashlight/pkg/speech/criterion/SequenceCriterion.h"

typedef std::map<std::string, unsigned int> Mapping;
namespace asr4real {
Mapping loadMapping(const std::string& filename);

/**
 * Perform a step of the MTL loss:
 * - find the corresponding label from the dataset for the given batchIdx
 * - apply the label classifier to the input features
 * - Return the categoricalCrossEntropy loss obtained from the results above
 *
 * @param enc_out : feature vector of dimension H x Time X Batch
 * @param crit : Linera classifier of dimension H x Nlabels
 * @param trainset : input dataset
 * @param i_map : Mapping from file id to file integer label
 * @param batchIdx : batch number
 *
 * @return : The loss vector, of dimension Batch x 1 x 1
 */
fl::Variable mtl_step(
    fl::Variable& enc_out,
    std::shared_ptr<fl::Linear> crit,
    std::shared_ptr<fl::Dataset> trainset,
    const Mapping& i_map,
    const int batchIdx);

/**
 * Map a file id to its corresponding label number with
 * fileID = "baseID#{label}"
 */
unsigned int getMapIndexFromFileID(
    const std::string& fileid,
    const Mapping& i_map);

/**
 * Extract the ID labels from a given batch
 * The labels are define as follow: file_id = "baseID#{label}"
 * @param  batch : batch extracted from a fl::Dataset.
 *                 We should be able to read each sample Id from
 *                 fl::app::asr::kSampleIdx
 * @param  i_map : mapping from label to label number
 * @param  batch_size
 *
 * @return An array X of shape (1, batch_size, 1) where X[0, a, 0] =
 *         label_number(batch.at(a))
 */
af::array buildIndexLabels(
    const std::vector<af::array>& batch,
    const Mapping& i_map,
    const int batch_size);
} // namespace asr4real
