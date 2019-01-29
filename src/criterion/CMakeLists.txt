cmake_minimum_required(VERSION 3.5.1)
# ----------------------------- Configuration -----------------------------
# Criterion backend - default to the AFML backend
set(W2L_CRITERION_BACKEND "CUDA" CACHE STRING "Backend with which to build criterion")
# Select from exactly one backend
set_property(CACHE W2L_CRITERION_BACKEND PROPERTY STRINGS CPU CUDA)
set(CRIT_BACKEND_USE_CPU OFF)
set(CRIT_BACKEND_USE_CUDA OFF)
if (W2L_CRITERION_BACKEND STREQUAL "CPU")
  set(CRIT_BACKEND_USE_CPU ON)
elseif (W2L_CRITERION_BACKEND STREQUAL "CUDA")
  set(CRIT_BACKEND_USE_CUDA ON)
elseif (W2L_CRITERION_BACKEND STREQUAL "OPENCL")
  set(CRIT_BACKEND_USE_OPENCL ON)
else ()
  message(FATAL_ERROR "Invalid criterion backend specified.")
endif ()

# ----------------------------- Dependencies -----------------------------
# Warp CTC
if (CRIT_BACKEND_USE_CUDA)
  # Build warpctc included in wav2letter/src/third_party/warpctc
  set(WARPCTC_DIR ${PROJECT_SOURCE_DIR}/src/third_party/warpctc)
  message(STATUS "Adding warpctc:")
  add_subdirectory(${WARPCTC_DIR} ${WARPCTC_DIR}/build)
  set(WARPCTC_INCLUDE_DIR ${WARPCTC_DIR}/include)
  # Download CUB for CUDA kernel primitives
  include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
endif()

# ----------------------------- Lib -----------------------------
set(
  CRITERION_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Seq2SeqCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FullConnectionCriterion.cpp
)

if (CRIT_BACKEND_USE_CUDA)  
  set(
    CRITERION_SOURCES
    ${CRITERION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/FullConnectionCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/ViterbiPath.cpp
    )

  # Compile CUDA kernels
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)
  # Setup
  set_cuda_cxx_compile_flags()
  set_cuda_arch_nvcc_flags()

  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
    )

  cuda_add_library(
    w2lCriterionCudaKernels
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/kernels/ViterbiPath.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/kernels/FullConnectionCriterion.cu
    )

  # Make sure CUB headers are present before building
  add_dependencies(w2lCriterionCudaKernels CUB)
elseif (CRIT_BACKEND_USE_CPU OR CRIT_BACKEND_USE_OPENCL)
  set(
    CRITERION_SOURCES
    ${CRITERION_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/FullConnectionCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/ViterbiPath.cpp
    )
endif ()

# Main library
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
  )

target_include_directories(
  criterion
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )

if (CRIT_BACKEND_USE_CUDA)
  target_link_libraries(
    criterion
    INTERFACE
    ${CUDA_LIBRARIES}
    warpctc # added directly from subdir
    w2lCriterionCudaKernels # cuda lib
    )
  
  target_include_directories(
    criterion
    INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${WARPCTC_INCLUDE_DIR}
    )
elseif (CRIT_BACKEND_USE_CPU)
  # OpenMP is already linked - see top-level CMakeLists.txt
endif()

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
