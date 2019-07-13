/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <arrayfire.h>
#include <flashlight/flashlight.h>

#include "common/FlashlightUtils.h"
#include "module/module.h"

using namespace fl;
using namespace w2l;

namespace {

std::string archDir = "";

} // namespace

TEST(W2lModuleTest, W2lSeqModule) {
  const std::string archfile = pathsConcat(archDir, "test_w2l_arch.txt");
  int nchannel = 4;
  int nclass = 40;
  int batchsize = 2;
  int inputsteps = 100;

  auto model = createW2lSeqModule(archfile, nchannel, nclass);

  auto input = af::randn(inputsteps, 1, nchannel, batchsize, f32);

  auto output = model->forward(noGrad(input));

  ASSERT_EQ(output.dims(), af::dim4(nclass, inputsteps, batchsize));

  batchsize = 1;
  input = af::randn(inputsteps, 1, nchannel, batchsize, f32);
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputsteps, batchsize));
}

TEST(W2lModuleTest, Serialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test.mdl";
  const std::string archfile = pathsConcat(archDir, "test_w2l_arch.txt");

  int C = 1, N = 5, B = 1, T = 10;
  auto model = createW2lSeqModule(archfile, C, N);

  auto input = noGrad(af::randn(T, 1, C, B, f32));
  auto output = model->forward(input);

  save(path, model);

  std::shared_ptr<Sequential> loaded;
  load(path, loaded);

  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded.get(), *model));
  ASSERT_TRUE(allClose(outputl, output));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for arch
#ifdef MODULE_TEST_ARCHDIR
  archDir = MODULE_TEST_ARCHDIR;
#endif

  return RUN_ALL_TESTS();
}
