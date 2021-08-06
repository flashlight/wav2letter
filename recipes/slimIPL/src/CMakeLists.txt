cmake_minimum_required(VERSION 3.10)

add_executable(
    train_slimipl
    Train.cpp
    MyLogger.cpp
  )

target_include_directories(
    train_slimipl
    PRIVATE
    ${PROJECT_SOURCE_DIR}
  )

target_link_libraries(
  train_slimipl
  PRIVATE
  flashlight::flashlight-app-asr
  flashlight::flashlight-app-common
  )
