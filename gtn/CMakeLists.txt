# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

cmake_minimum_required (VERSION 2.8)
cmake_policy(VERSION 2.8)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")

set(PKGNAME gtn)

add_library(${PKGNAME} MODULE
  GTN.c GTNBuffer.c GTNIterator.c GTNLogAdd.c GTNMax.c
)

install(TARGETS ${PKGNAME} LIBRARY
  DESTINATION ${LUA_CPATH})

FILE(GLOB luafiles *.lua)

install(FILES ${luafiles}
  DESTINATION ${LUA_PATH}/${PKGNAME})
