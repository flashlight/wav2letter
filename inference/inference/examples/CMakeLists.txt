cmake_minimum_required(VERSION 3.5.1)
project(StreamingInferenceExamples)
add_library(StreamingInferenceExamples INTERFACE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(util_example
  ${CMAKE_CURRENT_LIST_DIR}/Util.cpp
)

add_dependencies(util_example cereal)
add_dependencies(util_example streaming_inference_common)

target_include_directories(
  util_example
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${cereal_INCLUDE_DIRS}
    ${wav2letter-inference_SOURCE_DIR}
    ${wav2letter++_SOURCE_DIR}/src
)

target_link_libraries(
  util_example
  PUBLIC
    streaming_inference_common
)

# audio_to_words_example library is used by the example executables
add_library(audio_to_words_example
  ${CMAKE_CURRENT_LIST_DIR}/AudioToWords.cpp
)

target_include_directories(
  audio_to_words_example
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${wav2letter-inference_SOURCE_DIR}
)

target_link_libraries(
  audio_to_words_example
  PUBLIC
    util_example
    streaming_inference_modules_nn_backend
    streaming_inference_decoder
    decoder-library
)

function(build_example TARGET SRCFILE)
  message("Examples: add executable ${TARGET}")
  add_executable(${TARGET}
    ${SRCFILE}
  )
  add_dependencies(StreamingInferenceExamples ${TARGET})
  target_include_directories(
    ${TARGET}
    PRIVATE
      ${CMAKE_CURRENT_LIST_DIR}
      ${GFLAGS_INCLUDE_DIRS}
  )
  target_link_libraries(
    ${TARGET}
    PRIVATE
      streaming_inference_modules_nn_backend
      streaming_inference_modules_feature
      audio_to_words_example
      ${GFLAGS_LIBRARIES}
  )
endfunction(build_example)

build_example(simple_streaming_asr_example
  ${CMAKE_CURRENT_LIST_DIR}/SimpleStreamingASRExample.cpp)
build_example(Multithreaded_streaming_asr_example
  ${CMAKE_CURRENT_LIST_DIR}/MultithreadedStreamingASRExample.cpp)
