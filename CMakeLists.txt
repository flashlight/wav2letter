cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++)

# C++ 11 is required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(flashlight REQUIRED CONFIG)
if (NOT TARGET flashlight::flashlight-app-asr)
  message(FATAL_ERROR "flashlight must be build with app/asr for wav2letter++")
endif ()

add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
