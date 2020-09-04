cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-recipes-utilities-convlm_serializer)

add_executable(
    SerializeConvLM
    Serialize.cpp
    Utils.cpp
  )

target_include_directories(
    SerializeConvLM
    PUBLIC
    ${CMAKE_PROJECT_SOURCE}
  )

target_link_libraries(
  SerializeConvLM
  PUBLIC
  flashlight::flashlight
  flashlight::flashlight-app-asr
  ${GLOG_LIBRARIES}
  )

add_library(
  convlm-serializer
  INTERFACE
  )

target_sources(
  convlm-serializer
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  convlm-serializer
  INTERFACE
  flashlight::flashlight
  flashlight::flashlight-app-asr
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  convlm-serializer
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${CMAKE_PROJECT_SOURCE}
  )
