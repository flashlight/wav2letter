cmake_minimum_required(VERSION 3.5.1)

add_library(
  common
  INTERFACE
  )

target_sources(
  common
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Defines.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FlashlightUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Transforms.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  )

target_link_libraries(
  common
  INTERFACE
  flashlight::flashlight
  ${GFLAGS_LIBRARIES}
  )

target_include_directories(
  common
  INTERFACE
  ${FLASHLIGHT_INCLUDE_DIRS}
  ${GFLAGS_INCLUDE_DIRS}
  )
