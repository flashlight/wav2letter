/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "inference/module/nn/backend/fbgemm/PackedGemmMatrixFP16.h"

namespace w2l {
namespace streaming {

std::string debugString(
    const fbgemm::PackedGemmMatrixFP16& packedMatrix,
    bool dumpContent) {
  std::stringstream ss;
  ss << "PackedGemmMatrixFP16:{"
     << " num_rows:" << packedMatrix.numRows()
     << " ncol:" << packedMatrix.numCols()
     << " block_row_size:" << packedMatrix.blockRowSize()
     << " last_brlock_ow:" << packedMatrix.lastBrow()
     << " block_col_size:" << packedMatrix.blockColSize()
     << " num_block_row:" << packedMatrix.numBrow()
     << " num_clock_col:" << packedMatrix.numBcol()
     << " mat_size:" << packedMatrix.matSize();
  if (dumpContent) {
    ss << " content=\n";
    for (int r = 0; r < packedMatrix.numRows(); ++r) {
      for (int c = 0; c < packedMatrix.numCols(); ++c) {
        ss << fbgemm::cpu_half2float(packedMatrix.operator()(r, c)) << ", ";
      }
      ss << std::endl;
    }
  }
  ss << "}";
  return ss.str();
}

} // namespace streaming
} // namespace w2l
