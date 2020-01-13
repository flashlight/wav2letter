/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright (c) 2012 Jakob Progsch, Vaclav Zeman

#pragma once

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace w2l {
namespace streaming {
namespace example {

/**
* A simple C++11 Thread Pool implementation.
* Source - https://github.com/progschj/ThreadPool
*
* Basic usage:
  \code
    // create thread pool with 4 worker threads
    ThreadPool pool(4);

    // enqueue and store future
    auto result = pool.enqueue([](int answer) { return answer; }, 42);

    // get result from future
    std::cout << result.get() << std::endl;
  \endcode
*/
class ThreadPool {
 public:
  /**
   * the constructor just launches given amount of workers
   * \param [in] threads number of threads
   * \param [in] initFn initialization code (if any) that will be run on all the
   * threads
   */
  ThreadPool(
      size_t threads,
      const std::function<void(size_t)>& initFn = nullptr);

  /**
   * add new work item to the pool
   * \param [in] f function to be executed in threadpool
   * \param [in] args varadic arguments for the function
   */
  template <class F, class... Args>
  auto enqueue(F&& f, Args&&... args)
      -> std::future<typename std::result_of<F(Args...)>::type>;
  ///  destructor joins all threads.
  ~ThreadPool();

 private:
  // need to keep track of threads so we can join them
  std::vector<std::thread> workers;
  // the task queue
  std::queue<std::function<void()>> tasks;

  // synchronization
  std::mutex queue_mutex;
  std::condition_variable condition;
  bool stop;
}; // namespace flclassThreadPool

inline ThreadPool::ThreadPool(
    size_t threads,
    const std::function<void(size_t)>& initFn /* = nullptr */)
    : stop(false) {
  for (size_t id = 0; id < threads; ++id)
    workers.emplace_back([this, initFn, id] {
      if (initFn) {
        initFn(id);
      }
      for (;;) {
        std::function<void()> task;

        {
          std::unique_lock<std::mutex> lock(this->queue_mutex);
          this->condition.wait(
              lock, [this] { return this->stop || !this->tasks.empty(); });
          if (this->stop && this->tasks.empty())
            return;
          task = std::move(this->tasks.front());
          this->tasks.pop();
        }

        task();
      }
    });
}

template <class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type> {
  using return_type = typename std::result_of<F(Args...)>::type;

  auto task = std::make_shared<std::packaged_task<return_type()>>(
      std::bind(std::forward<F>(f), std::forward<Args>(args)...));

  std::future<return_type> res = task->get_future();
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    // don't allow enqueueing after stopping the pool
    if (stop)
      throw std::runtime_error("enqueue on stopped ThreadPool");

    tasks.emplace([task]() { (*task)(); });
  }
  condition.notify_one();
  return res;
}

inline ThreadPool::~ThreadPool() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }
  condition.notify_all();
  for (std::thread& worker : workers)
    worker.join();
  std::cout << "exit ThreadPool::~ThreadPool()\n";
}

} // namespace example
} // namespace streaming
} // namespace w2l
