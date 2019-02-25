cmake_minimum_required(VERSION 3.5.1)

add_library(
  data
  INTERFACE
  )

target_sources(
  data
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lListFilesDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lNumberedFilesDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/NumberedFilesLoader.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Featurize.cpp
  )

target_link_libraries(
  data
  INTERFACE
  common
  feature
  flashlight::flashlight
  ${GLOG_LIBRARIES}
  ${MKL_LIBRARIES}
  )

target_include_directories(
  data
  INTERFACE
  ${MKL_INCLUDE_DIR}
  )
