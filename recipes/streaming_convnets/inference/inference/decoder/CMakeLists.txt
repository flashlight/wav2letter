add_library(streaming_inference_decoder INTERFACE)

find_package(kenlm REQUIRED)

target_sources(streaming_inference_decoder
  INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/Decoder.cpp
)

get_target_property(DEC_SOURCES decoder-library INTERFACE_SOURCES)

target_link_libraries(
  streaming_inference_decoder
  INTERFACE
    ${KENLM_LIBRARIES}
    streaming_inference_common
    decoder-library
    wav2letter-libraries
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
