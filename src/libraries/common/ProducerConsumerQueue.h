/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace w2l {

/**
 * ProducerConsumerQueue implements a thread-safe queue for a certain type of
 * object, which support multiple producers generating objects and multiple
 * consumers consuming them at the same time. This queue makes sure that both
 * adding and getting methods are atomic.
 *
 * The overall design of the ProducerConsumerQueue following the principle of --
 * waking up as fewer threads as possible. Specifically, after adding an element
 * to the queue, an producer will wake up a single consumer and an single
 * producer if the queue is not full. Similarly, after consuming an element, an
 * consumer will wake up a single producer and and single comsumer if the queue
 * is not empty.
 *
 *
 * Sample usage:
 *
 *   ProducerConsumerQueue<T> queue();
 *
 *   // Producer threads
 *   for (thread in 0 ... N) {
 *       T obj = generate();
 *       queue.add(std::move(obj));
 *   }
 *   queue.finishAdding(); // when all producer threads joined
 *
 *   // Consumer threads
 *   for (thread in N + 1 ... M) {
 *       T obj;
 *       while(queue.get(obj)) {
 *           ...
 *       }
 *       // It's reasonable to break the loop when get() returns false, because
 *       // it means the queue is empty and adding is also finshed.
 *   }
 *
 */

template <typename T>
class ProducerConsumerQueue {
 public:
  explicit ProducerConsumerQueue(int maxSize = 3000)
      : maxSize_(maxSize), isAddingFinished_(false) {}

  /*
   * - Adds an element to the queue.
   * - Ignores the current one if adding is finished.
   * - Notifies another producer when queue is not full.
   * - Notifies a consumer.
   */
  void add(T unit) {
    std::unique_lock<std::mutex> lock(mutex_);
    producerCondition_.wait(
        lock, [this]() { return !isFull() || isAddingFinished_; });

    if (isAddingFinished_) {
      return;
    }
    queue_.push(std::move(unit));

    if (!isFull()) {
      producerCondition_.notify_one();
    }
    consumerCondition_.notify_one();
  }

  /*
   * - Pops an element from the queue.
   * - Returns false when adding is finished and queue is empty.
   * - Notifies another consumer when queue is not empty.
   * - Notifies a producer.
   */
  bool get(T& unit) {
    std::unique_lock<std::mutex> lock(mutex_);
    consumerCondition_.wait(
        lock, [this]() { return !isEmpty() || isAddingFinished_; });
    if (isEmpty()) {
      return false;
    }
    unit = std::move(queue_.front());
    queue_.pop();

    if (!isEmpty()) {
      consumerCondition_.notify_one();
    }
    producerCondition_.notify_one();

    return true;
  }

  /*
   * - Sets the status of the queue to be adding-finished.
   * - Notifies all the consumers to consume the remaining elements.
   */
  void finishAdding() {
    std::unique_lock<std::mutex> lock(mutex_);
    isAddingFinished_ = true;
    consumerCondition_.notify_all();
  }

  /*
   * - Clears the queue.
   * - Resets the status of the queue to be adding-unfinished.
   * - Notifies all the consumers and producers to work.
   */
  void clear() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (!isEmpty()) {
      queue_.pop();
    }
    isAddingFinished_ = false;

    producerCondition_.notify_all();
    consumerCondition_.notify_all();
  }

 private:
  std::condition_variable producerCondition_;
  std::condition_variable consumerCondition_;

  std::mutex mutex_;
  std::queue<T> queue_;
  int maxSize_;
  bool isAddingFinished_;

  bool isFull() const {
    return queue_.size() >= maxSize_;
  }

  bool isEmpty() const {
    return queue_.size() == 0;
  }
};

} // namespace w2l
