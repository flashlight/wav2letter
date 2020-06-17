cmake_minimum_required(VERSION 3.5.1)

add_library(
  augmentation
  INTERFACE
  )

target_sources(
  augmentation
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/AudioAugmenter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AudioLoader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/AdditiveNoise.cpp
  )

target_link_libraries(
  augmentation
  INTERFACE
  common
  data
  flashlight::flashlight
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  augmentation
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  )

add_subdirectory(${PROJECT_SOURCE_DIR}/src/experimental/augmentation/tests)
