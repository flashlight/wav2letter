/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>

#include <gtest/gtest.h>

#include "inference/common/DefaultMemoryManager.h"
#include "inference/common/IOBuffer.h"
#include "inference/common/MemoryManager.h"
#include "inference/module/ModuleProcessingState.h"

namespace w2l {
namespace streaming {

TEST(MemoryManager, Basic) {
  std::shared_ptr<MemoryManager> mm = std::make_shared<DefaultMemoryManager>();
  auto floatPtr = mm->makeShared<float>(100);
  auto intPtr = mm->makeShared<int>(100);

  EXPECT_NE(floatPtr, nullptr);
  EXPECT_NE(floatPtr, nullptr);
}
} // namespace streaming
} // namespace w2l
