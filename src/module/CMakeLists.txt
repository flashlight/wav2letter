cmake_minimum_required(VERSION 3.5.1)

add_library(
  module
  INTERFACE
  )

target_sources(
  module
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvLmModule.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SpecAugment.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TDSBlock.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lModule.cpp
  )

target_link_libraries(
  module
  INTERFACE
  common
  flashlight::flashlight
  ${cereal_LIBRARIES}
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  module
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )
