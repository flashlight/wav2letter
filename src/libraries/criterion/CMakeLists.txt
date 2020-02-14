cmake_minimum_required(VERSION 3.5.1)

add_library(
  criterion-library
  INTERFACE
  )

target_sources(
  criterion-library
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/ViterbiPath.cpp
  )

target_link_libraries(
  criterion-library
  INTERFACE
  common-library
  )

# ------------------------- CUDA-specific -------------------------

if (W2L_LIBRARIES_USE_CUDA)
  include(${CMAKE_MODULE_PATH}/BuildCUB.cmake)
  include(${CMAKE_MODULE_PATH}/CUDAUtils.cmake)

  set_cuda_cxx_compile_flags()
  set_cuda_arch_nvcc_flags()

  # hacky: add -fPIC to nvcc flags if needed
  if (W2L_BUILD_FOR_PYTHON OR CMAKE_POSITION_INDEPENDENT_CODE)
    cuda_enable_position_independent_code()
  endif ()

  cuda_include_directories(
    ${CUB_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    )

  # Qualify name because CUDA target is public
  cuda_add_library(
    w2l-criterion-library-cuda
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/CriterionUtils.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ForceAlignmentCriterion.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/FullConnectionCriterion.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ViterbiPath.cu
    )

  add_dependencies(w2l-criterion-library-cuda CUB)

  target_link_libraries(
    criterion-library
    INTERFACE
    ${CUDA_LIBRARIES}
    w2l-criterion-library-cuda
    )

  target_include_directories(
    criterion-library
    INTERFACE
    ${CUDA_INCLUDE_DIRS}
    )
endif ()
