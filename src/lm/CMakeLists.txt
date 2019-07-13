cmake_minimum_required(VERSION 3.5.1)

add_library(
  lm
  INTERFACE
  )

target_sources(
  lm
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvLM.cpp
  )

target_link_libraries(
  lm
  INTERFACE
  common
  lm-library
  )
