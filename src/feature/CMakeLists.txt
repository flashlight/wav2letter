cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Dependencies -----------------------------
# MKL
find_package(MKL REQUIRED)
if (MKL_FOUND)
  message(STATUS "MKL found")
else()
  message(STATUS "MKL not found")
endif()

# FFTW
find_package(FFTW REQUIRED)
if (FFTW_FOUND)
  message(STATUS "FFTW found")
else()
  message(FATAL_ERROR "FFTW not found")
endif()

# ----------------------------- Lib -----------------------------
add_library(
  feature
  INTERFACE
  )

target_sources(
  feature
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Ceplifter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Dct.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Derivatives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Dither.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Mfcc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Mfsc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PowerSpectrum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PreEmphasis.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/SpeechUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TriFilterbank.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Windowing.cpp
  )

target_link_libraries(
  feature
  INTERFACE
  common
  ${MKL_LIBRARIES}
  ${FFTW_LIBRARIES}
  )

target_include_directories(
  feature
  INTERFACE
  src/feature
  ${MKL_INCLUDE_DIR}
  )
