# Try to find the KenLM library
#
# The following variables are optionally searched for defaults
#  KENLM_DIR: Base directory where all KENLM components are found
#  KENLM_INCLUDE_DIR: Directory where KENLM headers are found
#  KENLM_LIB_DIR: Directory where KENLM library is found
#  KENLM_UTIL_LIB: Directory where KENLM utils are found
#  KENLM_MAX_ORDER: Max order build setting for kenlm
#
# The following are set after configuration is done:
#  KENLM_FOUND
#  KENLM_REQUIRED_HEADERS - only the needed headers
#  KENLM_LIBRARIES
#

message(STATUS "Looking for KenLM")

find_library(
  KENLM_LIB
  kenlm
  ENV KENLM_ROOT_DIR
  PATHS
    $ENV{KENLM_ROOT_DIR}
  HINT
    ${KENLM_DIR}
    ${KENLM_LIB_DIR}
    $ENV{KENLM_ROOT_DIR}/lib
    $ENV{KENLM_ROOT_DIR}/build/lib
    )

find_library(
  KENLM_UTIL_LIB
  kenlm_util
  ENV KENLM_ROOT_DIR
  PATHS
    $ENV{KENLM_ROOT_DIR}
  HINT
    ${KENLM_DIR}
    ${KENLM_LIB_DIR}
    $ENV{KENLM_ROOT_DIR}/lib
    $ENV{KENLM_ROOT_DIR}/build/lib
  )

if(KENLM_LIB)
  message(STATUS "Using kenlm library found in ${KENLM_LIB}")
else()
  message(FATAL_ERROR "kenlm library not found; please set CMAKE_LIBRARY_PATH or KENLM_LIB")
endif()

if(KENLM_UTIL_LIB)
  message(STATUS "Using kenlm utils library found in ${KENLM_LIB}")
else()
  message(FATAL_ERROR "kenlm utils library not found; please set CMAKE_LIBRARY_PATH or KENLM_UTIL_LIB")
endif()

# find a model header, then get the entire include directory. We need to do this because
# cmake consistently confuses other things along this path
find_file(KENLM_MODEL_HEADER
  lm/model.hh
  ENV KENLM_INC
  HINT
    ${KENLM_DIR}
    ${KENLM_DIR}/lm
    $ENV{KENLM_INC}
    ${KENLM_INC}
    $ENV{KENLM_ROOT_DIR}
    $ENV{KENLM_ROOT_DIR}/lm
    ${KENLM_LIB}
  PATHS
    ${KENLM_INC}
    $ENV{KENLM_INC}
    ${KENLM_ROOT_DIR}
    $ENV{KENLM_ROOT_DIR}
  )

if(KENLM_MODEL_HEADER)
  message(STATUS "kenlm lm/model.hh found in ${KENLM_MODEL_HEADER}")
else()
  message(FATAL_ERROR "kenlm lm/model.hh not found; please set CMAKE_INCLUDE_PATH or KENLM_INC")
endif()
get_filename_component(KENLM_INCLUDE_LM ${KENLM_MODEL_HEADER} DIRECTORY)
get_filename_component(KENLM_INCLUDE_DIR ${KENLM_INCLUDE_LM} DIRECTORY)

set(KENLM_LIBRARIES ${KENLM_LIB} ${KENLM_UTIL_LIB})
set(KENLM_INCLUDE_DIRS ${KENLM_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(kenlm DEFAULT_MSG KENLM_INCLUDE_DIRS KENLM_LIBRARIES)

if (kenlm_FOUND)
  message(STATUS "Found kenlm (include: ${KENLM_INCLUDE_DIRS}, library: ${KENLM_LIBRARIES})")
  mark_as_advanced(KENLM_ROOT_DIR KENLM_INCLUDE_DIRS KENLM_LIBRARIES)
endif()
