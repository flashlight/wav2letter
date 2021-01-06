# Try to find the KenLM library
#
# The following variables are optionally searched for defaults
#  KENLM_ROOT: Base directory where all KENLM components are found
#
# The following are set after configuration is done:
#  KENLM_FOUND
#  KENLM_LIBRARIES
#  KENLM_INCLUDE_DIRS
#  KENLM_INCLUDE_DIRS_LM
#

message(STATUS "Looking for KenLM")

# Required for KenLM to read ARPA files in compressed format
find_package(LibLZMA REQUIRED)
find_package(BZip2 REQUIRED)
find_package(ZLIB REQUIRED)

find_library(
  KENLM_LIB
  kenlm
  HINTS
    ${KENLM_ROOT}/lib
    ${KENLM_ROOT}/build/lib
  PATHS
    $ENV{KENLM_ROOT}/lib
    $ENV{KENLM_ROOT}/build/lib
  )

find_library(
  KENLM_UTIL_LIB
  kenlm_util
  HINTS
    ${KENLM_ROOT}/lib
    ${KENLM_ROOT}/build/lib
  PATHS
    $ENV{KENLM_ROOT}/lib
    $ENV{KENLM_ROOT}/build/lib
  )

if(KENLM_LIB)
  message(STATUS "Using kenlm library found in ${KENLM_LIB}")
else()
  message(STATUS "kenlm library not found; if you already have kenlm installed, please set CMAKE_LIBRARY_PATH, KENLM_LIB or KENLM_ROOT environment variable")
endif()

if(KENLM_UTIL_LIB)
  message(STATUS "Using kenlm utils library found in ${KENLM_UTIL_LIB}")
else()
  message(STATUS "kenlm utils library not found; if you already have kenlm installed, please set CMAKE_LIBRARY_PATH, KENLM_UTIL_LIB or KENLM_ROOT environment variable")
endif()

# find a model header, then get the entire include directory. We need to do this because
# cmake consistently confuses other things along this path
find_path(KENLM_MODEL_HEADER
  model.hh
  PATH_SUFFIXES
    kenlm/lm
    include/kenlm/lm
  HINTS
    ${KENLM_ROOT}/lm
    ${KENLM_ROOT}/include/kenlm/lm
  PATHS
    $ENV{KENLM_ROOT}/lm
    $ENV{KENLM_ROOT}/include/kenlm/lm
  )

if(KENLM_MODEL_HEADER)
  message(STATUS "kenlm model.hh found in ${KENLM_MODEL_HEADER}")

  get_filename_component(KENLM_INCLUDE_LM ${KENLM_MODEL_HEADER} DIRECTORY)
  get_filename_component(KENLM_INCLUDE_DIR ${KENLM_INCLUDE_LM} DIRECTORY)
else()
  message(STATUS "kenlm model.hh not found; if you already have kenlm installed, please set CMAKE_INCLUDE_PATH, KENLM_MODEL_HEADER or KENLM_ROOT environment variable")
endif()

set(KENLM_LIBRARIES
  ${KENLM_LIB}
  ${KENLM_UTIL_LIB}
  ${LIBLZMA_LIBRARIES}
  ${BZIP2_LIBRARIES}
  ${ZLIB_LIBRARIES}
)
# Some KenLM include paths are relative to [include dir]/kenlm, not just [include dir] (bad)
set(KENLM_INCLUDE_DIRS_LM ${KENLM_INCLUDE_LM})
set(KENLM_INCLUDE_DIRS ${KENLM_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(kenlm DEFAULT_MSG KENLM_INCLUDE_DIRS KENLM_LIBRARIES)

if (kenlm_FOUND)
  message(STATUS "Found kenlm (include: ${KENLM_INCLUDE_DIRS}, library: ${KENLM_LIBRARIES})")
  mark_as_advanced(KENLM_ROOT KENLM_INCLUDE_DIRS KENLM_LIBRARIES)
endif()
