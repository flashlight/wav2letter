add_library(streaming_inference_decoder INTERFACE)

target_sources(streaming_inference_decoder
  INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/Decoder.cpp
)

target_link_libraries(
  streaming_inference_decoder
  INTERFACE
    ${KENLM_LIBRARIES}
    streaming_inference_common
    flashlight::fl-libraries
)

target_compile_definitions(
  streaming_inference_decoder
  INTERFACE
   KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
)

target_include_directories(
  streaming_inference_decoder
  INTERFACE
    ${KENLM_INCLUDE_DIRS}
)
