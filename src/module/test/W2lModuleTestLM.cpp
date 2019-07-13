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

#include <common/FlashlightUtils.h>
#include <module/module.h>

using namespace fl;
using namespace w2l;

namespace {

std::string archDir = "";

} // namespace

TEST(W2lModuleTest, GCNN14BAdaptiveSoftmax) {
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_as.txt");
  int nclass = 221452;
  int batchsize = 2;
  int inputlength = 100;
  std::vector<int> tail = {10000, 50000, 200000, nclass};

  auto model = createW2lSeqModule(archfile, 1, nclass);
  auto as = std::make_shared<fl::AdaptiveSoftMax>(4096, tail);
  auto criterion = std::make_shared<fl::AdaptiveSoftMaxLoss>(as);
  model->eval();
  criterion->eval();
  auto input = af::range(af::dim4(inputlength, batchsize), f32);
  auto output = model->forward(noGrad(input));
  output = as->forward(output);

  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));

  // batchsize = 1
  batchsize = 1;
  input = af::range(af::dim4(inputlength), f32);
  output = model->forward(noGrad(input));
  output = as->forward(output);
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));
}

TEST(W2lModuleTest, GCNN14BCrossEntropy) {
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_ce.txt");
  int nclass = 30;
  int batchsize = 2;
  int inputlength = 100;

  auto model = createW2lSeqModule(archfile, 1, nclass);
  model->eval();
  auto input = af::range(af::dim4(inputlength, batchsize), f32);
  auto output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));

  // batchsize = 1
  batchsize = 1;
  input = af::range(af::dim4(inputlength), f32);
  output = model->forward(noGrad(input));
  ASSERT_EQ(output.dims(), af::dim4(nclass, inputlength, batchsize));
}

TEST(W2lModuleTest, SerializationGCNN14BAdaptiveSoftmax) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test.mdl";
  const std::string archfile = pathsConcat(archDir, "gcnn_14B_lm_arch_as.txt");

  int nclass = 221452;
  int batchsize = 2;
  int inputlength = 10;
  std::vector<int> tail = {10000, 50000, 200000, nclass};

  std::shared_ptr<fl::Module> model = createW2lSeqModule(archfile, 1, nclass);
  auto as = std::make_shared<fl::AdaptiveSoftMax>(4096, tail);
  std::shared_ptr<BinaryModule> criterion =
      std::make_shared<fl::AdaptiveSoftMaxLoss>(as);
  model->eval();
  criterion->eval();
  auto input = noGrad(af::range(af::dim4(inputlength, batchsize), f32));
  auto output = model->forward({input})[0];
  auto output_criterion =
      std::dynamic_pointer_cast<AdaptiveSoftMaxLoss>(criterion)
          ->getActivation()
          ->forward(output);

  save(path, model, criterion);

  std::shared_ptr<Module> loaded_model;
  std::shared_ptr<BinaryModule> loaded_criterion;
  load(path, loaded_model, loaded_criterion);

  auto outputl = loaded_model->forward({input})[0];
  auto outputl_criterion =
      std::dynamic_pointer_cast<AdaptiveSoftMaxLoss>(loaded_criterion)
          ->getActivation()
          ->forward(output);

  ASSERT_TRUE(allParamsClose(*loaded_model.get(), *model));
  ASSERT_TRUE(allParamsClose(*loaded_criterion.get(), *criterion));
  ASSERT_TRUE(allClose(outputl, output));
  ASSERT_TRUE(allClose(outputl_criterion, output_criterion));
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  // Resolve directory for arch
#ifdef MODULE_TEST_ARCHDIR
  archDir = MODULE_TEST_ARCHDIR;
#endif

  return RUN_ALL_TESTS();
}
