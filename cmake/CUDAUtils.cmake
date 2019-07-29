# CUDAUtils - general utilities for working with CUDA
#
# This file should only be included if CUDA is to be linked (e.g. CUDA is the specified
# criterion backend), as it searches for CUDA.
#

### Find CUDA
find_package(CUDA 9.2 QUIET) # CUDA 9.2 is required for >= ArrayFire 3.6.1
if (CUDA_FOUND)
  message(STATUS "CUDA found (library: ${CUDA_LIBRARIES} include: ${CUDA_INCLUDE_DIRS})")
else()
  message(FATAL_ERROR "CUDA required to build CUDA criterion backend")
endif()

# This line must be placed after find_package(CUDA)
include(${CMAKE_MODULE_PATH}/select_compute_arch.cmake)

### Set compilation flags
# NVCC doesn't properly listen to cxx version flags, so manually override.
# This MUST be done after CUDA is found, but before any cuda libs/binaries have
# been created.
function (set_cuda_cxx_compile_flags)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-std=c++11" PARENT_SCOPE)
  # Using host flags makes things bad - keep things clean
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)
endfunction()

### Detect GPU architectures
# Detect architectures (see select_compute_arch.cmake) and
# add appropriate flags to nvcc for gencode/arch/ptx
function (set_cuda_arch_nvcc_flags)
  set(
    CUDA_architecture_build_targets
    "Common"
    CACHE STRING "Detected CUDA architectures for this build"
    )
  cuda_select_nvcc_arch_flags(cuda_arch_flags ${CUDA_architecture_build_targets})
  message(STATUS "CUDA architecture flags: " ${cuda_arch_flags})
  # Add to flag list
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${cuda_arch_flags}" PARENT_SCOPE)
  mark_as_advanced(CUDA_architecture_build_targets)
endfunction()

function (cuda_enable_position_independent_code)
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xcompiler;-fPIC" PARENT_SCOPE)
endfunction ()


### Taken from FindCUDA.cmake
# A bug with how cuda_add_library is written causes linking a CUDA library
# with other libraries to fail since no scope specifier is given for the link.
# Reimplemented here, and explicitly denote the link (PRIVATE).
# https://gitlab.kitware.com/cmake/cmake/issues/16602
function(cuda_add_library cuda_target)
  cuda_add_cuda_include_once()

  # Separate the sources from the options
  cuda_get_sources_and_options(_sources _cmake_options _options ${ARGN})
  cuda_build_shared_library(_cuda_shared_flag ${ARGN})
  # Create custom commands and targets for each file.
  cuda_wrap_srcs( ${cuda_target} OBJ _generated_files ${_sources}
    ${_cmake_options} ${_cuda_shared_flag}
    OPTIONS ${_options} )

  # Compute the file name of the intermedate link file used for separable
  # compilation.
  cuda_compute_separable_compilation_object_file_name(link_file ${cuda_target} "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  # Add the library.
  add_library(${cuda_target} ${_cmake_options}
    ${_generated_files}
    ${_sources}
    ${link_file}
    )

  # Add a link phase for the separable compilation if it has been enabled.  If
  # it has been enabled then the ${cuda_target}_SEPARABLE_COMPILATION_OBJECTS
  # variable will have been defined.
  cuda_link_separable_compilation_objects("${link_file}" ${cuda_target} "${_options}" "${${cuda_target}_SEPARABLE_COMPILATION_OBJECTS}")

  target_link_libraries(${cuda_target}
      PRIVATE ${CUDA_LIBRARIES}
    )

  # We need to set the linker language based on what the expected generated file
  # would be. CUDA_C_OR_CXX is computed based on CUDA_HOST_COMPILATION_CPP.
  set_target_properties(${cuda_target}
    PROPERTIES
    LINKER_LANGUAGE ${CUDA_C_OR_CXX}
    )

endfunction()
