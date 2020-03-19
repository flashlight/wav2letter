/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/cereal.hpp>
#include <cereal/details/traits.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/vector.hpp>
#include <fbgemm/FbgemmFP16.h>

namespace cereal {

template <typename Archive>
void save(
    Archive& ar,
    const std::shared_ptr<fbgemm::PackedGemmMatrixFP16>& packedMatrix) {
  const uint nElements = packedMatrix->matSize();
  std::vector<fbgemm::float16> tempBuf(nElements);
  // PackedGemmMatrixFP16::unpack() does not change the state of the object's
  // state, however, it is not marked const. Thus casting off the const here.
  fbgemm::PackedGemmMatrixFP16* nonConstPackedMatrix =
      const_cast<fbgemm::PackedGemmMatrixFP16*>(packedMatrix.get());
  nonConstPackedMatrix->unpack(
      tempBuf.data(), fbgemm::matrix_op_t::NoTranspose);

  ar(packedMatrix->numRows(),
     packedMatrix->numCols(),
     packedMatrix->blockRowSize(),
     packedMatrix->lastBrow(),
     packedMatrix->blockColSize(),
     packedMatrix->numBrow(),
     packedMatrix->numBcol(),
     packedMatrix->matSize());
  ar(tempBuf);
}

template <typename Archive>
void load(
    Archive& ar,
    std::shared_ptr<fbgemm::PackedGemmMatrixFP16>& packedMatrix) {
  int numRows = 0;
  int numCols = 0;

  // The following params are unused but kept for backward compatibility
  int blockRowSize = 0;
  int lastBrow = 0;
  int blockColSize = 0;
  int numBrow = 0;
  int numBcol = 0;
  int matSize = 0;

  ar(numRows,
     numCols,
     blockRowSize,
     lastBrow,
     blockColSize,
     numBrow,
     numBcol,
     matSize);

  std::vector<fbgemm::float16> tempBufFp16;
  ar(tempBufFp16);

  std::vector<float> tempBufFp32(numRows * numCols);
  for (int i = 0; i < numRows * numCols; ++i) {
    tempBufFp32[i] = fbgemm::cpu_half2float(tempBufFp16[i]);
  }

  constexpr float alpha = 1.0;
  packedMatrix = std::make_shared<fbgemm::PackedGemmMatrixFP16>(
      fbgemm::matrix_op_t::NoTranspose,
      numRows,
      numCols,
      alpha,
      tempBufFp32.data());
}

} // namespace cereal

namespace w2l {
namespace streaming {

std::string debugString(
    const fbgemm::PackedGemmMatrixFP16& packedMatrix,
    bool dumpContent = false);

}
} // namespace w2l
