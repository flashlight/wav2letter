cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Dependencies -----------------------------
# CBLAS
if (W2L_LIBRARIES_USE_MKL)
  find_package(MKL REQUIRED)
  set(CBLAS_LIBRARIES ${MKL_LIBRARIES})
  set(CBLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
else ()
  find_package(CBLAS REQUIRED)
endif ()
message(STATUS "CBLAS found (include: ${CBLAS_INCLUDE_DIR}, library: ${CBLAS_LIBRARIES})")

# FFTW
find_package(FFTW REQUIRED)
if (FFTW_FOUND)
  message(STATUS "FFTW found")
else()
  message(FATAL_ERROR "FFTW not found")
endif()

# ----------------------------- Lib -----------------------------
add_library(
  feature-library
  INTERFACE
  )

target_sources(
  feature-library
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
  feature-library
  INTERFACE
  ${CBLAS_LIBRARIES}
  ${FFTW_LIBRARIES}
  )

target_include_directories(
  feature-library
  INTERFACE
  src/libraries/feature
  ${CBLAS_INCLUDE_DIR}
  )
