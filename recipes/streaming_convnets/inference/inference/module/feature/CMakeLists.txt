add_library(streaming_inference_modules_feature
  ${CMAKE_CURRENT_LIST_DIR}/LogMelFeature.cpp
)

add_dependencies(streaming_inference_modules_feature cereal)

target_link_libraries(
  streaming_inference_modules_feature
  PUBLIC
    ${cereal_LIBRARIES}
    streaming_inference_common
    flashlight::fl-libraries
)

target_include_directories(
  streaming_inference_modules_feature
  PUBLIC
    ${cereal_INCLUDE_DIRS}
    # going up one directory to use #include "inference/*/*.h"
    ${streaming_inference_SOURCE_DIR}
)
