cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # no compiler extensions like gnu++11

# All include directories are based on namespace structure and use ./src
# as their base directory
include_directories(${CMAKE_SOURCE_DIR}/src)

# ----------------------------- Configuration -----------------------------

# Build tests
option(BUILD_TESTS "Build tests for wav2letter++" ON)
if (BUILD_TESTS)
  enable_testing()
  add_subdirectory(${CMAKE_SOURCE_DIR}/src/tests)
endif ()

# Build examples
option(BUILD_EXAMPLES "Build examples for wav2letter++" ON)

# ------------------------ Global External Dependencies ------------------------
# ArrayFire
find_package(ArrayFire 3.6.1 REQUIRED)
if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
  if (AFML_USE_UNIFIED)
    # Set the AF_PATH environment variable to wherever the ArrayFire libs are
    # located so they can be loaded in the unified backend
    set(ENV{AF_PATH} ${ArrayFire_LIBRARIES})
  endif ()
else()
  message(FATAL_ERROR "ArrayFire not found")
endif()

# Find GLog
find_package(GLOG REQUIRED)
if (GLOG_FOUND)
  message(STATUS "GLOG found")
else()
  message(FATAL_ERROR "GLOG not found")
endif()

# Find GFlags
find_package(GFLAGS REQUIRED)
if (GFLAGS_FOUND)
  message(STATUS "GFLAGS found")
else()
  message(FATAL_ERROR "GFLAGS not found")
endif()

# Find and setup OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
  message(STATUS "OpenMP found")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}"
  )
else()
  message(STATUS "OpenMP not found - building without OpenMP")
endif()

# ------------------------ flashlight ------------------------
find_package(flashlight REQUIRED)
if (flashlight_FOUND)
  message("flashlight found (include: ${FLASHLIGHT_INCLUDE_DIRS} lib: flashlight::flashlight )")
  if (NOT TARGET flashlight::Distributed)
    message(FATAL_ERROR "flashlight must be build in distributed mode or wav2letter++")
  else ()
    message(STATUS "flashlight built in distributed mode.")
  endif ()
else()
  message(FATAL_ERROR "flashlight not found.")
endif ()

# ------------------------ Components ------------------------

# Common
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/common)

# Feature
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/feature)

# Runtime
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/runtime)

# Module
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/module)

# Decoder
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/decoder)

# Criterion
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/criterion)

# Data
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/data)

# ----------------------------- wav2letter++ lib  -----------------------------

add_library(
  wav2letter++
  ""
  )

set_target_properties(
  wav2letter++
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  wav2letter++
  PUBLIC
  common
  criterion
  data
  decoder
  module
  runtime
  )

target_include_directories(
  wav2letter++
  PUBLIC
  ${FLASHLIGHT_INCLUDE_DIRS} # includes are <flashlight/...>
  )

# ----------------------------- Train -----------------------------
add_executable(
  Train
  Train.cpp
)

target_link_libraries(
  Train
  wav2letter++
  )

# ----------------------------- Test -----------------------------
add_executable(
  Test
  Test.cpp
)

target_link_libraries(
  Test
  wav2letter++
  )

# ----------------------------- Decoder -----------------------------
add_executable(
  Decoder
  Decode.cpp
)

target_link_libraries(
  Decoder
  wav2letter++
  )
