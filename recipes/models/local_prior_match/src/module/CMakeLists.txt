cmake_minimum_required(VERSION 3.5.1)

add_library(
  module_lpm_oss
  INTERFACE
  )

target_sources(
  module_lpm_oss
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LMWrapper.cpp
  )

target_link_libraries(
  module_lpm_oss
  INTERFACE
  module # inherit all properties from base interface
  )

target_include_directories(
  module_lpm_oss
  INTERFACE
  ${PROJECT_SOURCE_DIR}/../../..
  )
