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

  auto res_module1 = Residual(3);
  res_module1.add(conv);
  res_module1.add(bn);
  res_module1.add(relu);
  res_module1.addShortcut(1, 3);

  auto output1 = res_module1.forward(input);
  auto output1_true = relu.forward(output_bn + output_conv);
  ASSERT_TRUE(allClose(output1, output1_true));

  auto res_module2 = Residual(3);
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

TEST(ModuleTest, ResidualSerialization) {
  char* user = getenv("USER");
  std::string userstr = "unknown";
  if (user != nullptr) {
    userstr = std::string(user);
  }
  const std::string path = "/tmp/" + userstr + "_test_res";

  std::shared_ptr<Residual> model = std::make_shared<Residual>(3);
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

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
