# Try to find libsndfile
#
# Provides the cmake config target
# - SndFile::sndfile
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

# We require sndfile having been built with external libs.
find_package(Ogg REQUIRED)
find_package(Vorbis REQUIRED)
find_package(FLAC REQUIRED)

find_package(SndFile CONFIG)

if (TARGET SndFile::sndfile AND NOT SndFile_WITH_EXTERNAL_LIBS)
  message(WARNING "Found sndfile but was NOT built with external libs.")
endif()

if (NOT TARGET SndFile::sndfile)
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

  add_library(SndFile::sndfile UNKNOWN IMPORTED)
  set_target_properties(SndFile::sndfile PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIRS}"
    IMPORTED_LOCATION "${SNDFILE_LIBRARIES}"
    )
  set_property(
    TARGET SndFile::sndfile
    PROPERTY INTERFACE_LINK_LIBRARIES
    Vorbis::vorbis
    Vorbis::vorbisenc
    Ogg::ogg
    FLAC::FLAC
    )

  if (SNDFILE_FOUND)
    message(STATUS "Found libsndfile: (lib: ${SNDFILE_LIBRARIES} include: ${SNDFILE_INCLUDE_DIRS}")
  else()
    message(STATUS "libsndfile not found.")
  endif()
endif() # NOT TARGET SndFile::sndfile
