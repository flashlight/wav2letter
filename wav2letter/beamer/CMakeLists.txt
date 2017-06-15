cmake_minimum_required (VERSION 2.8)

SET(CMAKE_C_FLAGS "-std=c99")

ADD_DEFINITIONS(-DBMR_USE_QUICKSELECT)
ADD_DEFINITIONS(-DKENLM_MAX_ORDER=6)

FIND_LIBRARY(kenlmlib kenlm
  ENV KENLM_LIB
)

FIND_LIBRARY(kenlmutillib kenlm_util
  ENV KENLM_UTIL_LIB
)

IF(NOT kenlmlib)
  MESSAGE(FATAL_ERROR "kenlm library not found; please set CMAKE_LIBRARY_PATH or KENLM_LIB")
ENDIF()
MESSAGE(STATUS "Using kenlm library ${kenlmlib}")

FIND_FILE(kenlmmodel lm/model.hh
  ENV KENLM_INC)

IF(NOT kenlmmodel)
  MESSAGE(FATAL_ERROR "kenlm lm/model.hh not found; please set CMAKE_INCLUDE_PATH or KENLM_INC")
ENDIF()
GET_FILENAME_COMPONENT(kenlminc ${kenlmmodel} DIRECTORY)
GET_FILENAME_COMPONENT(kenlminc ${kenlminc} DIRECTORY)
MESSAGE(STATUS "Using kenlm include path ${kenlminc}")
INCLUDE_DIRECTORIES(${kenlminc})

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

find_package(Torch REQUIRED)

set(src
  lib/BMRArray.c
  lib/BMRBuffer.c
  lib/BMRTrie.c
  lib/BMRLM.cc
  lib/BMRDecoder.c
)

set(luasrc
  init.lua
  env.lua
  trie.lua
  lm.lua
  decoder.lua
)

add_torch_package(beamer "${src}" "${luasrc}")
target_link_libraries(beamer ${kenlmlib} ${kenlmutillib})
