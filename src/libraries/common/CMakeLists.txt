cmake_minimum_required(VERSION 3.5.1)

add_library(
  common-library
  INTERFACE
  )

target_sources(
  common-library
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Dictionary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/WordUtils.cpp
  )
