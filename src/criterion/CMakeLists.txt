cmake_minimum_required(VERSION 3.5.1)
# ----------------------------- Configuration -----------------------------
# Criterion backend - default to the AFML backend
set(CRITERION_BACKEND "CUDA" CACHE STRING "Backend with which to build criterion")
# Select from exactly one backend
set_property(CACHE CRITERION_BACKEND PROPERTY STRINGS CPU CUDA)
set(CRIT_BACKEND_USE_CPU OFF)
set(CRIT_BACKEND_USE_CUDA OFF)
if (CRITERION_BACKEND STREQUAL "CPU")
  set(CRIT_BACKEND_USE_CPU ON)
elseif (CRITERION_BACKEND STREQUAL "CUDA")
  set(CRIT_BACKEND_USE_CUDA ON)
elseif (CRITERION_BACKEND STREQUAL "OPENCL")
  set(CRIT_BACKEND_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "Invalid criterion backend specified.")
endif ()

# ----------------------------- Dependencies -----------------------------
# Only the CUDA criterion backend has distinct external dependencies
# CUDA
find_package(CUDA 9.2 QUIET) # CUDA 9.2 is required for >= ArrayFire 3.6.1
if (CUDA_FOUND)
  message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
else()
  message(STATUS "CUDA not found")
  if (AFML_USE_CUDA)
    message(FATAL_ERROR "CUDA required to build CUDA criterion backend")
  endif ()
endif()

# Warp CTC
if (CRIT_BACKEND_USE_CUDA)
  # Build warpctc included in wav2letter/src/third_party/warpctc
  set(WARPCTC_DIR ${PROJECT_SOURCE_DIR}/src/third_party/warpctc)
  message(STATUS "Adding warpctc:")
  add_subdirectory(${WARPCTC_DIR} ${WARPCTC_DIR}/build)
  set(WARPCTC_INCLUDE_DIR ${WARPCTC_DIR}/include)
endif()

# ----------------------------- Lib -----------------------------
set(
  CRITERION_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ConnectionistTemporalCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Seq2SeqCriterion.cpp
)

if (CRIT_BACKEND_USE_CUDA)
  set(
    CRITERION_SOURCES
    ${CRITERION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/ConnectionistTemporalCriterion.cpp
  )
elseif (CRIT_BACKEND_USE_CPU OR CRIT_BACKEND_USE_OPENCL)
  set(
    CRITERION_SOURCES
    ${CRITERION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/ConnectionistTemporalCriterion.cpp
  )
endif ()

# Criterion main
add_library(
  criterion
  INTERFACE
  )

target_sources(
  criterion
  INTERFACE
  ${CRITERION_SOURCES}
  )

target_link_libraries(
  criterion
  INTERFACE
  attention
  window
  common
  flashlight::flashlight
  ${GLOG_LIBRARIES}
  ${cereal_LIBRARIES}
  ArrayFire::afcuda
  ${CUDA_LIBRARIES}
  warpctc # added directly from subdir
  )

target_include_directories(
  criterion
  INTERFACE
  src
  src/criterion
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${ArrayFire_INCLUDE_DIRS}
  ${WARPCTC_INCLUDE_DIR}
  )

# ---------------------------- Attention/Window -----------------------------
# Attention
add_library(
  attention
  INTERFACE
  )

target_sources(
  attention
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/ContentAttention.cpp
  )

target_include_directories(
  attention
  INTERFACE
  attention
  src/flashlight
  ${cereal_INCLUDE_DIRS}
  )

target_link_libraries(
  attention
  INTERFACE
  flashlight::flashlight
  common
  ${cereal_LIBRARIES}
  )

# Window
add_library(
  window
  INTERFACE
  )

target_sources(
  window
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/MedianWindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/SoftWindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/StepWindow.cpp
  )

target_include_directories(
  window
  INTERFACE
  attention
  src/flashlight
  ${cereal_INCLUDE_DIRS}
  )

target_link_libraries(
  window
  INTERFACE
  flashlight::flashlight
  common
  ${cereal_LIBRARIES}
  )
