cmake_minimum_required(VERSION 3.10)

add_executable(
    SerializeConvLM
    Serialize.cpp
    Utils.cpp
  )

target_include_directories(
    SerializeConvLM
    PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

target_link_libraries(
  SerializeConvLM
  PRIVATE
  flashlight::flashlight-app-asr
  )
