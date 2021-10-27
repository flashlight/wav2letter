cmake_minimum_required(VERSION 3.10)

project(wav2letter++ CXX C)

# C++ 14 is required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(flashlight CONFIG REQUIRED)
if (FL_USE_CUDA)
  enable_language(CUDA)
endif()

if (NOT TARGET flashlight::fl_pkg_speech)
  message(FATAL_ERROR "flashlight must be build with app/asr for wav2letter++")
endif ()

include(CheckCXXCompilerFlag)
# All libraries should have their symbols exported so plugins can lazily
# symbols from any of them
check_cxx_compiler_flag("-rdynamic" COMPILER_SUPPORTS_RDYNAMIC)
if(${COMPILER_SUPPORTS_RDYNAMIC})
  message(STATUS "-rdynamic supported.")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -rdynamic")
else()
  message(WARNING
    "This compiler doesn't support dynamic symbol exports. "
    "Plugin functionality likely won't work.")
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
