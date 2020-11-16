cmake_minimum_required(VERSION 3.5.1)

add_executable(
    SerializeConvLM
    Serialize.cpp
    Utils.cpp
  )

target_include_directories(
    SerializeConvLM
    PUBLIC
    ${PROJECT_SOURCE_DIR}
  )

target_link_libraries(
  SerializeConvLM
  PUBLIC
  flashlight::flashlight-app-asr
  )
