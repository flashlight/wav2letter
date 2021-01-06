cmake_minimum_required(VERSION 3.10)

# Creates build rules for streaming inference

project(streaming_inference CXX C)

option(STREAMING_INFERENCE_BUILD_TESTS
  "Build tests for streaming_inference"    ON)
option(STREAMING_INFERENCE_BUILD_EXAMPLES
  "Build examples for streaming_inference" ON)
option(STREAMING_INFERENCE_BUILD_TOOLS
  "Build converter tools for streaming_inference" ON)
set(KENLM_MAX_ORDER 6 CACHE
    STRING
    "Maximum ngram order for KenLM"
)

#--- Find flashlight ---
find_package(flashlight CONFIG REQUIRED)
if (NOT TARGET flashlight::fl-libraries)
  message(FATAL_ERROR "flashlight::fl-libraries must be built for streaming inference")
endif ()

# Inference backend names MUST match paths along
# inference/module/nn/backend/{backend_name}
set(AVAILABLE_INFERENCE_BACKENDS fbgemm)
set(STREAMING_INFERENCE_BACKEND fbgemm CACHE
  STRING
  "Inference backend library"
)
set_property(
  CACHE
  STREAMING_INFERENCE_BACKEND
  PROPERTY STRINGS
  ${AVAILABLE_INFERENCE_BACKENDS}
)

# C++ 14 is required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(streaming_inference "")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

#--- Find GFlags ---
find_package(gflags REQUIRED)

#--- Find/build Cereal ---
# If cereal is found in a user-defined location, use it rather than
# downloading from source
find_package(cereal)
if (NOT TARGET cereal AND NOT cereal_FOUND)
  message(STATUS "cereal NOT found. Will download from source")
  set(CEREAL_INSTALL_PATH ${FL_INSTALL_INC_DIR}/cereal)
  include(${CMAKE_MODULE_PATH}/BuildCereal.cmake)
  # Move cereal headers at install time
  install(DIRECTORY ${CEREAL_SOURCE_DIR}/include/cereal
    DESTINATION ${CEREAL_INSTALL_PATH}
    COMPONENT cereal
    FILES_MATCHING
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN ".git" EXCLUDE
    )
  install(FILES ${CEREAL_SOURCE_DIR}/LICENSE ${CEREAL_SOURCE_DIR}/README.md
    DESTINATION ${CEREAL_INSTALL_PATH}
    )
endif()

# ------------------ add subdirectories --------------------

# Creates interface library: streaming_inference_common
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/common)

# Creates interface library: streaming_inference_decoder
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/decoder)

# Creates interface library: streaming_inference_modules
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/inference/module)

# Creates executable targets foreach example.
if (STREAMING_INFERENCE_BUILD_EXAMPLES)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/examples)
endif()

# ------------------------ Tests ------------------------

# Creates executable targets foreach example.
if (STREAMING_INFERENCE_BUILD_TESTS)
  enable_testing()
  include(TestUtils)
  # add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/test) // FIXME
endif()

# ------------------------ Tools ------------------------
if (STREAMING_INFERENCE_BUILD_TOOLS)
  add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/tools)
endif()

# -------------- streaming_inference top level target -----------------

set_target_properties(
  streaming_inference
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 14
)

target_include_directories(
  streaming_inference
  PUBLIC
    ${streaming_inference_SOURCE_DIR}
)

target_link_libraries(
  streaming_inference
  PUBLIC
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
)

install(TARGETS streaming_inference
  EXPORT streaming_inference-export
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include/inference
)
