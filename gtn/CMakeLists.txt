#
# (c) 2015 Facebook. All rights reserved.
# Author: Ronan Collobert <locronan@fb.com>
#
#

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
