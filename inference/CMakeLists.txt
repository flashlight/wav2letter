cmake_minimum_required(VERSION 3.5.1)

# Creates build rules for wav2letter inference processing graph.

include(GNUInstallDirs)

option(W2L_INFERENCE_BUILD_TESTS
  "Build tests for wav2letter-inference"    ON)
option(W2L_INFERENCE_BUILD_EXAMPLES
  "Build examples for wav2letter-inference" ON)
set(KENLM_MAX_ORDER 6 CACHE
    STRING
    "Maximum ngram order for KenLM"
)

# Inference backend names MUST match paths along
# inference/module/nn/backend/{backend_name}
set(AVAILABLE_INFERENCE_BACKENDS fbgemm)
set(W2L_INFERENCE_BACKEND fbgemm CACHE
  STRING
  "Inference backend library"
)
set_property(
  CACHE
  W2L_INFERENCE_BACKEND
  PROPERTY STRINGS
  ${AVAILABLE_INFERENCE_BACKENDS}
)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(wav2letter-inference)

add_library(wav2letter-inference "")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

#--- Find GFlags ---
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

#--- Download and build Google test ---
# Only if we're not building w2l tests elsewhere, else we'll
# define the same custom target twice with ExternalProject_add
include(BuildGoogleTest)

#--- Download/build Cereal ---
set(CEREAL_INSTALL_PATH ${CMAKE_CURRENT_LIST_DIR}/cereal)
include(Buildcereal)
# Move cereal headers
install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
  DESTINATION ${CEREAL_INSTALL_PATH}
  COMPONENT cereal
  FILES_MATCHING
  PATTERN "*.hpp"
  PATTERN "*.h"
  PATTERN ".git" EXCLUDE
)

install(
  FILES
    ${CEREAL_SOURCE_DIR}/LICENSE
    ${CEREAL_SOURCE_DIR}/README.md
  DESTINATION
    ${CEREAL_INSTALL_PATH}
)

# ------------------ add subdirectories --------------------

# Creates interface library: streaming_inference_common
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/common)

# Creates interface library: streaming_inference_decoder
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/decoder)

# Creates interface library: streaming_inference_modules
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/module)

# Creates executable targets foreach example.
if (W2L_INFERENCE_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/examples)
endif()

# ------------------------ Tests ------------------------

add_custom_target(inference_tests)

# Builds a single test source file.
function(build_test SRCFILE)
  get_filename_component(target ${SRCFILE} NAME_WE)
  set(target inference_${target})
  message(STATUS "Tests: add executable ${target}")
  add_executable(${target} ${SRCFILE})
  add_dependencies(wav2letter-inference ${target})
  add_dependencies(${target} gtest) # make sure gtest is built first
  add_dependencies(inference_tests ${target})
  target_link_libraries(
    ${target}
    PRIVATE
    ${GTEST_LIBRARIES}
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
    Threads::Threads
  )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${GTEST_INCLUDE_DIR}
    # going up one directory to use #include "inference/*/*.h"
    ${wav2letter-inference_SOURCE_DIR}
    ${W2L_INFERENCE_TESTS_PATH}
    ${cereal_INCLUDE_DIRS}
  )
  add_test(${target} ${target})
endfunction(build_test)

set(W2L_INFERENCE_TESTS_PATH
  ${wav2letter-inference_SOURCE_DIR}/inference/module/test)

set(W2L_INFERENCE_TESTS_SOURCES
  ${W2L_INFERENCE_TESTS_PATH}/Conv1dTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/IdentityTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/LayerNormTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/LinearTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/LogMelFeatureTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/MemoryManagerTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/ReluTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/ResidualTest.cpp
  ${W2L_INFERENCE_TESTS_PATH}/TDSBlockTest.cpp
)

if (W2L_INFERENCE_BUILD_TESTS)
  find_package(Threads REQUIRED)
  enable_testing()
  foreach(SRC_FILE ${W2L_INFERENCE_TESTS_SOURCES})
    build_test(${SRC_FILE})
  endforeach()
endif()

# -------------- wav2letter-inference top level target -----------------

set_target_properties(
  wav2letter-inference
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
)

target_include_directories(
  wav2letter-inference
  PUBLIC
    ${wav2letter-inference_SOURCE_DIR}
)

target_link_libraries(
  wav2letter-inference
  PUBLIC
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
)

install(TARGETS wav2letter-inference
  EXPORT wav2letter-inference-export
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/wav2letter/inference
)
