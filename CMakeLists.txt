cmake_minimum_required(VERSION 3.10)

project(wav2letter++)

# C++ 11 is required
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(flashlight CONFIG REQUIRED)
if (NOT TARGET flashlight::flashlight-app-asr)
  message(FATAL_ERROR "flashlight must be build with app/asr for wav2letter++")
endif ()

add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
