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

# Sndfile
# The SndFile build/library doesn't correctly link targets for
# its Vorbis and FLAC dependencies, so they need to be manually found.
set(SNDFILE_DEP_LIBRARIES "")
set(SNDFILE_DEP_INCLUDE_DIRS "")
function(findAndCheckSndFileDependency DEP)
  find_package(${DEP} REQUIRED)
  if (${DEP}_FOUND)
    message(STATUS "Required SndFile dependency ${DEP} found.")
    set(SNDFILE_DEP_LIBRARIES ${SNDFILE_DEP_LIBRARIES} ${${DEP}_LIBRARIES})
    set(SNDFILE_DEP_INCLUDE_DIRS ${SNDFILE_DEP_INCLUDE_DIRS} ${${DEP}_INCLUDE_DIRS})
  else()
    message(FATAL_ERROR "Required SndFile dependency ${DEP} not found.")
  endif()
endfunction(findAndCheckSndFileDependency)
findAndCheckSndFileDependency(Ogg REQUIRED)
findAndCheckSndFileDependency(Vorbis REQUIRED)
findAndCheckSndFileDependency(VorbisEnc REQUIRED)
findAndCheckSndFileDependency(FLAC REQUIRED)
# Sndfile uses imported targets
find_package(SNDFILE REQUIRED)
if (SNDFILE_FOUND)
  message(STATUS "libsndfile found.")
else ()
  message(FATAL_ERROR "Required dependency libsndfile not found.")
endif ()

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
  ${CMAKE_CURRENT_SOURCE_DIR}/Sound.cpp
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
  ${SNDFILE_LIBRARIES}
  ${SNDFILE_DEP_LIBRARIES}
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  feature
  INTERFACE
  src/feature
  ${MKL_INCLUDE_DIR}
  ${SNDFILE_INCLUDE_DIRS}
  ${SNDFILE_DEP_INCLUDE_DIRS}
  )
