/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <flashlight/distributed/distributed.h>

#include "common/Defines.h"
#include "runtime/Distributed.h"

namespace w2l {

void initDistributed(
    int worldRank,
    int worldSize,
    int maxDevicesPerNode,
    const std::string& rndvFilepath) {
  if (rndvFilepath.empty()) {
    distributedInit(
        fl::DistributedInit::MPI,
        -1, // unused for MPI
        -1, // unused for MPI
        {{fl::DistributedConstants::kMaxDevicePerNode,
          std::to_string(maxDevicesPerNode)}});
  } else {
    distributedInit(
        fl::DistributedInit::FILE_SYSTEM,
        worldRank,
        worldSize,
        {{fl::DistributedConstants::kMaxDevicePerNode,
          std::to_string(maxDevicesPerNode)},
         {fl::DistributedConstants::kFilePath, rndvFilepath}});
  }
}

} // namespace w2l
