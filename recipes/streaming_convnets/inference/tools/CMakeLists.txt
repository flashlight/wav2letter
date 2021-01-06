cmake_minimum_required(VERSION 3.10)

add_executable(streaming_tds_model_converter StreamingTDSModelConverter.cpp)

if (NOT TARGET flashlight::flashlight-app-asr)
  message(FATAL_ERROR "flashlight must be build with app/asr for w2l inference tools")
endif ()

if (@FL_USE_CUDA@)
  find_dependency(CUDNN 7.1)
endif()

target_link_libraries(
    streaming_tds_model_converter
    PRIVATE
    streaming_inference_common
    streaming_inference_modules
    flashlight::flashlight-app-asr
)

target_include_directories(
    streaming_tds_model_converter
    PRIVATE
    ${PROJECT_SOURCE_DIR}
)
