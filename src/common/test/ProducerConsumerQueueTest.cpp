/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <future>
#include <mutex>
#include <string>
#include <vector>

#include "libraries/common/ProducerConsumerQueue.h"

using namespace w2l;

TEST(DictionaryTest, SingleThread) {
  ProducerConsumerQueue<int> queue(10);

  // Producing
  for (int i = 1; i <= 5; i++) {
    queue.add(i);
  }
  queue.finishAdding();

  // Consuming
  std::vector<int> output;
  int element;
  while (queue.get(element)) {
    output.emplace_back(element);
  }

  // Check
  ASSERT_THAT(output, testing::ElementsAre(1, 2, 3, 4, 5));
}

TEST(DictionaryTest, MultiThreads) {
  const int nElements = 1000, targetSum = 499500;
  const int nProducer = std::thread::hardware_concurrency() / 2,
            nConsumer = std::thread::hardware_concurrency() / 2;
  std::vector<int> consumerResults(nConsumer, 0);

  ProducerConsumerQueue<int> queue(nElements);

  // Define producer and consumers
  auto produce = [nElements, nProducer, &queue](int tid) {
    for (int i = tid; i < nElements; i += nProducer) {
      queue.add(i);
    }
  };

  auto consume = [&consumerResults, &queue](int tid) {
    int element;
    while (queue.get(element)) {
      consumerResults[tid] += element;
    }
  };

  // Run Test
  std::vector<std::future<void>> producerFutures(nConsumer);
  for (int i = 0; i < nProducer; i++) {
    producerFutures[i] = std::async(std::launch::async, produce, i);
  }

  std::vector<std::future<void>> consumerFutures(nConsumer);
  for (int i = 0; i < nConsumer; i++) {
    consumerFutures[i] = std::async(std::launch::async, consume, i);
  }

  for (int i = 0; i < nConsumer; i++) {
    producerFutures[i].wait();
  }
  queue.finishAdding();

  for (int i = 0; i < nConsumer; i++) {
    consumerFutures[i].wait();
  }

  // Check
  int predictSum = 0;
  for (const auto& element : consumerResults) {
    predictSum += element;
  }
  ASSERT_EQ(predictSum, targetSum);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  return RUN_ALL_TESTS();
}
