# Try to find libsndfile
#
# Inputs:
#   SNDFILE_INC_DIR: include directory for sndfile headers
#   SNDFILE_LIB_DIR: directory containing sndfile libraries
#   SNDFILE_ROOT_DIR: directory containing sndfile installation
#
# Defines:
#  SNDFILE_FOUND - system has libsndfile
#  SNDFILE_INCLUDE_DIRS - the libsndfile include directory
#  SNDFILE_LIBRARIES - Link these to use libsndfile
#

find_path(
  SNDFILE_INCLUDE_DIR
    sndfile.h
  PATHS
  ${SNDFILE_INC_DIR}
  ${SNDFILE_ROOT_DIR}/include
  PATH_SUFFIXES
  include
  )

find_library(
  SNDFILE_LIBRARY
  sndfile
  PATHS
  ${SNDFILE_LIB_DIR}
  ${SNDFILE_ROOT_DIR}
  PATH_SUFFIXES
  lib
  HINTS
  SNDFILE
  )

set(SNDFILE_INCLUDE_DIRS
  ${SNDFILE_INCLUDE_DIR}
  )
set(SNDFILE_LIBRARIES
  ${SNDFILE_LIBRARY}
  )

mark_as_advanced(SNDFILE_INCLUDE_DIRS SNDFILE_LIBRARIES)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SNDFILE DEFAULT_MSG SNDFILE_INCLUDE_DIRS GLOG_LIBRARIES)

if (SNDFILE_FOUND)
  message(STATUS "Found libsndfile: (lib: ${SNDFILE_LIBRARIES} include: ${SNDFILE_INCLUDE_DIRS}")
else()
  message(STATUS "libsndfile not found.")
endif()
