cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++)

# ----------------------------- Setup -----------------------------
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ----------------------------- Configuration -----------------------------

option(W2L_BUILD_LIBRARIES_ONLY "Build wav2letter-libraries only" OFF)
option(W2L_BUILD_INFERENCE
  "Build inference even when W2L_BUILD_LIBRARIES_ONLY is set" OFF)
option(W2L_LIBRARIES_USE_CUDA "Use CUDA in libraries-only build" ON)
option(W2L_LIBRARIES_USE_KENLM "Use KenLM in libraries-only build" ON)
option(W2L_LIBRARIES_USE_MKL "Use MKL in libraries-only build" ON)
option(W2L_BUILD_FOR_PYTHON "Build Python bindings" OFF)
option(W2L_BUILD_TESTS "Build tests for wav2letter++" ON)
option(W2L_BUILD_EXAMPLES "Build examples for wav2letter++" ON)
option(W2L_BUILD_EXPERIMENTAL "Build internal experimental components for wav2letter++" OFF)
option(W2L_BUILD_SCRIPTS "Build internal scripts for wav2letter++" OFF)
option(W2L_BUILD_RECIPES "Build recipes" ON)
option(W2L_BUILD_TOOLS "Build audio tools" OFF)
set(KENLM_MAX_ORDER 6 CACHE STRING "Maximum ngram order for KenLM")

# ------------------------- Dependency Fallback -------------------------

if (W2L_LIBRARIES_USE_MKL)
  find_package(MKL)
  if (NOT MKL_FOUND)
    message(WARNING "MKL not found; forcing W2L_LIBRARIES_USE_MKL=OFF.")
    set(W2L_LIBRARIES_USE_MKL OFF CACHE BOOL "" FORCE)
  endif ()
endif ()

# ------------------------- Libraries-Only Build ------------------------------

function (build_w2l_libraries)
  set(W2L_COMPILE_DEFINITIONS
    $<$<BOOL:${W2L_LIBRARIES_USE_CUDA}>:W2L_LIBRARIES_USE_CUDA>
    $<$<BOOL:${W2L_LIBRARIES_USE_KENLM}>:W2L_LIBRARIES_USE_KENLM>
    $<$<BOOL:${W2L_LIBRARIES_USE_MKL}>:W2L_LIBRARIES_USE_MKL>
    )
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/libraries)
  if (W2L_BUILD_FOR_PYTHON)
    add_subdirectory(${PROJECT_SOURCE_DIR}/bindings/python)
  endif ()
endfunction ()

if (W2L_BUILD_LIBRARIES_ONLY)
  build_w2l_libraries()
  if (W2L_BUILD_INFERENCE)
    add_subdirectory(${PROJECT_SOURCE_DIR}/inference)
  endif()
  return()
endif ()

# ------------------------ Global External Dependencies ------------------------
# ArrayFire
# The correct ArrayFire backend target is transitively included by flashlight
find_package(ArrayFire 3.6.1 REQUIRED)
if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
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
  message(STATUS "flashlight found (include: ${FLASHLIGHT_INCLUDE_DIRS} lib: flashlight::flashlight )")
  if (NOT TARGET flashlight::Distributed)
    message(FATAL_ERROR "flashlight must be build in distributed mode for wav2letter++")
  else ()
    message(STATUS "flashlight built in distributed mode.")
  endif ()
  if (NOT TARGET flashlight::Contrib)
    message(FATAL_ERROR "flashlight must be built with contrib features for wav2letter++")
  else ()
    message(STATUS "flashlight built with contrib features.")
  endif ()
else()
  message(FATAL_ERROR "flashlight not found.")
endif ()

# "Import" flashlight backend variables using the dummy target
if (TARGET flashlight::BACKEND_UNIFIED)
  set(FLASHLIGHT_BACKEND "UNIFIED")
  set(FLASHLIGHT_USE_UNIFIED ON)
elseif (TARGET flashlight::BACKEND_CPU)
  set(FLASHLIGHT_BACKEND "CPU")
  set(FLASHLIGHT_USE_CPU ON)
elseif (TARGET flashlight::BACKEND_CUDA)
  set(FLASHLIGHT_BACKEND "CUDA")
  set(FLASHLIGHT_USE_CUDA ON)
elseif (TARGET flashlight::BACKEND_OPENCL)
  set(FLASHLIGHT_BACKEND "OPENCL")
  set(FLASHLIGHT_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "could not determine flashlight backend")
endif ()

# ------------------ Configure Full Build ----------------------

set(W2L_LIBRARIES_USE_CUDA ${FLASHLIGHT_USE_CUDA} CACHE BOOL "" FORCE)
set(W2L_LIBRARIES_USE_KENLM ON CACHE BOOL "" FORCE)

# ------------------------ Components ------------------------

# wav2letter-libraries
build_w2l_libraries()

# Common
add_subdirectory(${PROJECT_SOURCE_DIR}/src/common)

# Criterion
add_subdirectory(${PROJECT_SOURCE_DIR}/src/criterion)

# Data
add_subdirectory(${PROJECT_SOURCE_DIR}/src/data)

# Module
add_subdirectory(${PROJECT_SOURCE_DIR}/src/module)

# Runtime
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime)

# Tests
if (W2L_BUILD_TESTS)
  enable_testing()
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/tests)
endif ()

if (W2L_BUILD_INFERENCE)
  add_subdirectory(${PROJECT_SOURCE_DIR}/inference)
endif()

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
  wav2letter-libraries
  common
  criterion
  data
  module
  runtime
  )

target_include_directories(
  wav2letter++
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src # all includes are based at src/
  ${FLASHLIGHT_INCLUDE_DIRS} # includes are <flashlight/...>
  )

# Build experimental assets if specified
if (W2L_BUILD_EXPERIMENTAL)
  message(STATUS "Building experimental assets.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/src/experimental)
endif ()

# Build scripts if specified
if (W2L_BUILD_SCRIPTS)
  message(STATUS "Building scripts.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/scripts)
endif ()

# Recipes for research
if (W2L_BUILD_RECIPES)
  message(STATUS "Building recipes.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
endif ()

# Tools for audio data and text
if (W2L_BUILD_TOOLS)
  message(STATUS "Building tools.")
  add_subdirectory(${PROJECT_SOURCE_DIR}/tools)
endif ()

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
