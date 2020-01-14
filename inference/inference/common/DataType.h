/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdio>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace w2l {

// Make sure enums support std::hash. Defect in C++ 11 that's fixed in C++ 14 -
// add a temporary backporting functor below.
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60970
// http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#2148
namespace cpp {

struct EnumHash {
  template <typename T>
  std::size_t operator()(T t) const {
    return static_cast<std::size_t>(t);
  }
};

template <typename Key_t>
using EnumHashType = typename std::
    conditional<std::is_enum<Key_t>::value, EnumHash, std::hash<Key_t>>::type;

template <typename Key_t, typename Value_t>
using enum_unordered_map =
    std::unordered_map<Key_t, Value_t, EnumHashType<Key_t>>;

} // namespace cpp

namespace streaming {

enum class DataType : uint32_t {
  UNINITIALIZED,
  FLOAT,
  FLOAT16,
};

bool dataTypeIsValid(DataType type);

const std::string dataTypeString(DataType type);

int dataTypeNumberOfBytes(DataType type);

} // namespace streaming
} // namespace w2l
