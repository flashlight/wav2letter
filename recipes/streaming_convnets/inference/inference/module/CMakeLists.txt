cmake_minimum_required(VERSION 3.5.1)

# Creates interface library: streaming_inference_modules_nn
include(${CMAKE_CURRENT_LIST_DIR}/nn/CMakeLists.txt)
include(${CMAKE_CURRENT_LIST_DIR}/feature/CMakeLists.txt)

add_library(streaming_inference_modules INTERFACE)

target_sources(streaming_inference_modules
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/InferenceModule.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ModuleParameter.cpp
  ${CMAKE_CURRENT_LIST_DIR}/ModuleProcessingState.cpp
)

add_dependencies(streaming_inference_modules streaming_inference_modules_nn)
add_dependencies(streaming_inference_modules cereal)

target_link_libraries(
    streaming_inference_modules
    INTERFACE
    ${cereal_LIBRARIES}
    streaming_inference_modules_nn
    streaming_inference_modules_feature
)

target_include_directories(
  streaming_inference_modules
  INTERFACE
  # going up one directory to use #include "inference/*/*.h"
  ${wav2letter-inference_SOURCE_DIR}
  ${cereal_INCLUDE_DIRS}
  ${nn_backend_INCLUDE_DIRS}
)
