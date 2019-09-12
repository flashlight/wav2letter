cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-recipes-models-utilities-convlm_serializer)

add_executable(
    SerializeConvLM
    Serialize.cpp
    Utils.cpp
  )

target_include_directories(
    SerializeConvLM
    PUBLIC
    ${PROJECT_SOURCE_DIR}/../../../..
  )

target_link_libraries(
  SerializeConvLM
  PUBLIC
  wav2letter++
  runtime
  common
  module
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
  module
  common
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  convlm-serializer
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/../../../..
  )
