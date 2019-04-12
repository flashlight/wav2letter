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

#include "common/Utils.h"
#include "module/module.h"

using namespace fl;
using namespace w2l;

TEST(ModuleTest, ResidualFwd) {
  auto conv = Conv2D(30, 50, 9, 7, 2, 3, 3, 2);
  auto bn = BatchNorm(2, 50);
  auto relu = ReLU();

  int batchsize = 10;
  auto input = Variable(af::randu(120, 100, 30, batchsize), false);

  auto output_conv = conv.forward(input);
  auto output_bn = bn.forward(output_conv);

  auto res_module1 = Residual();
  res_module1.add(conv);
  res_module1.add(bn);
  res_module1.add(relu);
  res_module1.addShortcut(1, 3);

  auto output1 = res_module1.forward(input);
  auto output1_true = relu.forward(output_bn + output_conv);
  ASSERT_TRUE(allClose(output1, output1_true));

  auto res_module2 = Residual();
  res_module2.add(conv);
  res_module2.add(bn);
  res_module2.add(relu);
  res_module2.addShortcut(1, 4);
  res_module2.addShortcut(1, 3);
  res_module2.addShortcut(2, 4);

  auto output2 = res_module2.forward(input);
  auto output2_true =
      relu.forward(output_bn + output_conv) + output_bn + output_conv;
  ASSERT_TRUE(allClose(output2, output2_true));
}

TEST(ModuleTest, ResidualFwdWithProjection) {
  auto linear1 = Linear(12, 8);
  auto relu1 = ReLU();
  auto linear2 = Linear(8, 4);
  auto relu2 = ReLU();
  auto linear3 = Linear(4, 4);
  auto relu3 = ReLU();
  auto projection1 = Linear(8, 4);
  auto projection2 = Linear(12, 4);

  auto input = Variable(af::randu(12, 10, 3, 4), false);
  auto output_true1 = linear1.forward(input);
  auto output_true = relu1.forward(output_true1);
  output_true = linear2.forward(output_true * 0.3);
  output_true =
      relu2.forward((output_true + projection1.forward(output_true1)) * 0.24);
  output_true = (output_true + projection2.forward(input)) * 0.5;
  output_true = linear3.forward(output_true);
  output_true = relu3.forward(output_true) + output_true;

  auto res_module = Residual();
  res_module.add(linear1);
  res_module.add(relu1);
  res_module.add(linear2);
  res_module.addScale(3, 0.3);
  res_module.add(relu2);
  res_module.addShortcut(1, 4, projection1);
  res_module.addScale(4, 0.24);
  res_module.add(linear3);
  res_module.addShortcut(0, 5, projection2);
  res_module.addScale(5, 0.5);
  res_module.add(relu3);
  res_module.addShortcut(5, 7);

  auto output_res = res_module.forward(input);

  ASSERT_TRUE(allClose(output_res, output_true));
}

TEST(ModuleTest, ResidualSerialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test_res";

  std::shared_ptr<Residual> model = std::make_shared<Residual>();
  model->add(Linear(12, 6));
  model->add(Linear(6, 6));
  model->add(ReLU());
  model->addShortcut(1, 3);
  save(path, model);

  std::shared_ptr<Residual> loaded;
  load(path, loaded);

  auto input = Variable(af::randu(12, 10, 3, 4), false);
  auto output = model->forward(input);
  auto outputl = loaded->forward(input);

  ASSERT_TRUE(allParamsClose(*loaded.get(), *model));
  ASSERT_TRUE(allClose(outputl, output));
}

TEST(ModuleTest, TDSFwd) {
  int batchsize = 10;
  int timesteps = 120;
  int w = 4;
  int c = 10;

  auto tds = TDSBlock(c, 9, w);
  auto input = Variable(af::randu(timesteps, w, c, batchsize), false);

  auto output = tds.forward({input})[0];

  ASSERT_EQ(output.dims(0), timesteps);
  ASSERT_EQ(output.dims(1), w);
  ASSERT_EQ(output.dims(2), c);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
