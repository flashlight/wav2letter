cmake_minimum_required(VERSION 3.10)

add_library(streaming_inference_common
  ${CMAKE_CURRENT_LIST_DIR}/DataType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/DefaultMemoryManager.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Functions.cpp
  ${CMAKE_CURRENT_LIST_DIR}/IOBuffer.cpp
)

add_dependencies(streaming_inference_common cereal)

target_include_directories(
  streaming_inference_common
  PUBLIC
    ${cereal_INCLUDE_DIRS}
    # going up one directory to use #include "inference/*/*.h"
    ${streaming_inference_SOURCE_DIR}
)
