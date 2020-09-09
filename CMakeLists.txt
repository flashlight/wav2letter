cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++)

# C++ 11 is required
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# ArrayFire
# The correct ArrayFire backend target is transitively included by flashlight
find_package(ArrayFire 3.7.1 REQUIRED)
if (ArrayFire_FOUND)
  message(STATUS "ArrayFire found (include: ${ArrayFire_INCLUDE_DIRS}, library: ${ArrayFire_LIBRARIES})")
else()
  message(FATAL_ERROR "ArrayFire not found")
endif()

find_package(flashlight REQUIRED)
if (flashlight_FOUND)
  message(STATUS "flashlight found (include: ${FLASHLIGHT_INCLUDE_DIRS} lib: flashlight::flashlight )")
  if (NOT TARGET flashlight::flashlight-app-asr)
    message(FATAL_ERROR "flashlight must be build with app/asr for wav2letter++")
  else ()
    message(STATUS "flashlight/app/asr is found.")
  endif ()
  if (NOT TARGET flashlight::fl-libraries)
    message(FATAL_ERROR "flashlight must be build with lib for wav2letter++")
  else ()
    message(STATUS "flashlight/lib is found.")
  endif ()
endif ()


add_subdirectory(${PROJECT_SOURCE_DIR}/recipes)
