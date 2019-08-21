cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Dependencies -----------------------------

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
  data
  INTERFACE
  )

target_sources(
  data
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/Featurize.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ListFileDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Sound.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lBlobsDataset.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/W2lListFilesDataset.cpp
  )

target_link_libraries(
  data
  INTERFACE
  common
  feature-library
  flashlight::flashlight
  ${GLOG_LIBRARIES}
  ${MKL_LIBRARIES}
  ${SNDFILE_LIBRARIES}
  ${SNDFILE_DEP_LIBRARIES}
  )

target_include_directories(
  data
  INTERFACE
  ${MKL_INCLUDE_DIR}
  ${SNDFILE_INCLUDE_DIRS}
  ${SNDFILE_DEP_INCLUDE_DIRS}
  )
