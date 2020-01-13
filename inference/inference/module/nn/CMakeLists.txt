if (NOT W2L_INFERENCE_BACKEND)
  message(FATAL_ERROR "W2L_INFERENCE_BACKEND must be defined."
    " supported values={FBGEMMM, }")
endif()

# Creates target streaming_inference_modules_nn_backend
# Appends CMAKE_MODULE_PATH
# Sets BACKEND_FOUND=true
include(${CMAKE_CURRENT_LIST_DIR}/backend/${W2L_INFERENCE_BACKEND}/CMakeLists.txt)

if (NOT ${BACKEND_FOUND})
  message(FATAL_ERROR
  "W2L_INFERENCE_BACKEND:${W2L_INFERENCE_BACKEND} is not found.")
endif()

add_library(streaming_inference_modules_nn INTERFACE)

add_library(streaming_inference_modules_nn_impl
  ${CMAKE_CURRENT_LIST_DIR}/Conv1d.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Identity.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LayerNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Linear.cpp
  ${CMAKE_CURRENT_LIST_DIR}/LocalNorm.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Relu.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Residual.cpp
  ${CMAKE_CURRENT_LIST_DIR}/Sequential.cpp
  ${CMAKE_CURRENT_LIST_DIR}/TDSBlock.cpp
)

target_include_directories(
  streaming_inference_modules_nn_impl
  PUBLIC
    ${cereal_INCLUDE_DIRS}
    # going up one directory to use #include "inference/*/*.h"
    ${wav2letter-inference_SOURCE_DIR}/..
)

add_dependencies(streaming_inference_modules_nn_impl cereal)

target_link_libraries(
  streaming_inference_modules_nn_impl
  PUBLIC
    streaming_inference_modules_nn_backend
    streaming_inference_common
)

add_dependencies(streaming_inference_modules_nn cereal)

target_link_libraries(
  streaming_inference_modules_nn
  INTERFACE
    streaming_inference_modules_nn_impl
    streaming_inference_modules_nn_backend
    streaming_inference_common
)

add_dependencies(streaming_inference_modules_nn cereal)
