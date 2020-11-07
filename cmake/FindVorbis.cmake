# - Find vorbis
#
# Provides Vorbis config targets:
# - Vorbis::vorbis
# - Vorbis::vorbisenc
#
# Find the native vorbis and  vorbisenc includes and libraries
#
#  VORBIS_INCLUDE_DIRS - where to find vorbis.h, etc.
#  VORBIS_LIBRARIES    - List of libraries when using vorbis.
#  VORBIS_FOUND        - True if vorbis found.
#
#  VORBISENC_INCLUDE_DIRS - where to find vorbisenc.h, etc.
#  VORBISENC_LIBRARIES    - List of libraries when using vorbisenc.
#  VORBISENC_FOUND        - True if vorbisenc found.


find_package(Vorbis CONFIG)

if (NOT TARGET Vorbis::vorbis OR NOT TARGET Vorbis::vorbisenc)
  if (VORBIS_INCLUDE_DIR)
	  # Already in cache, be silent
	  set(VORBIS_FIND_QUIETLY TRUE)
  endif ()

  if (VORBISENC_INCLUDE_DIR)
    # Already in cache, be silent
    set(VORBISENC_FIND_QUIETLY TRUE)
  endif ()

  find_package(Ogg QUIET)

  find_package(PkgConfig QUIET)
  pkg_check_modules(PC_VORBIS QUIET vorbis)
  pkg_check_modules(PC_VORBISENC QUIET vorbisenc)

  set(VORBIS_VERSION ${PC_VORBIS_VERSION})

  find_path(VORBIS_INCLUDE_DIR vorbis/codec.h
    HINTS
      ${PC_VORBIS_INCLUDEDIR}
      ${PC_VORBIS_INCLUDE_DIRS}
      ${VORBIS_ROOT}
    )

  find_library(VORBIS_LIBRARY
    NAMES
      vorbis
      vorbis_static
      libvorbis
      libvorbis_static
    HINTS
      ${PC_VORBIS_LIBDIR}
      ${PC_VORBIS_LIBRARY_DIRS}
      ${VORBIS_ROOT}
    )

  # Handle the QUIETLY and REQUIRED arguments and set VORBIS_FOUND
  # to TRUE if all listed variables are TRUE.
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(Vorbis
    REQUIRED_VARS
      VORBIS_LIBRARY
      VORBIS_INCLUDE_DIR
      OGG_FOUND
    VERSION_VAR
          VORBIS_VERSION
    )

  if (VORBIS_FOUND)
    set(VORBIS_INCLUDE_DIRS ${VORBIS_INCLUDE_DIR})
    set(VORBIS_LIBRARIES ${VORBIS_LIBRARY} ${OGG_LIBRARIES})
      if (NOT TARGET Vorbis::vorbis)
      add_library(Vorbis::vorbis UNKNOWN IMPORTED)
      set_target_properties(Vorbis::vorbis PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${VORBIS_INCLUDE_DIR}"
        IMPORTED_LOCATION "${VORBIS_LIBRARY}"
        INTERFACE_LINK_LIBRARIES Ogg::ogg
      )
    endif ()
  endif ()

  mark_as_advanced(VORBIS_INCLUDE_DIR VORBIS_LIBRARY)

  set(VORBISENC_VERSION ${PC_VORBISENC_VERSION})

  find_path(VORBISENC_INCLUDE_DIR vorbis/vorbisenc.h
    HINTS
      ${PC_VORBISENC_INCLUDEDIR}
      ${PC_VORBISENC_INCLUDE_DIRS}
      ${VORBISENC_ROOT}
    )

  find_library(VORBISENC_LIBRARY
    NAMES
      vorbisenc
      vorbisenc_static
      libvorbisenc
      libvorbisenc_static
    HINTS
      ${PC_VORBISENC_LIBDIR}
      ${PC_VORBISENC_LIBRARY_DIRS}
      ${VORBISENC_ROOT}
    )

  # Handle the QUIETLY and REQUIRED arguments and set VORBISENC_FOUND
  # to TRUE if all listed variables are TRUE.
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(VorbisEnc
    REQUIRED_VARS
      VORBISENC_LIBRARY
      VORBISENC_INCLUDE_DIR
      VORBIS_FOUND
    VERSION_VAR
          VORBISENC_VERSION
    )

  if (VORBISENC_FOUND)
    set(VORBISENC_INCLUDE_DIRS ${VORBISENC_INCLUDE_DIR})
    set(VORBISENC_LIBRARIES ${VORBISENC_LIBRARY} ${VORBIS_LIBRARIES})
      if (NOT TARGET Vorbis::vorbisenc)
      add_library(Vorbis::vorbisenc UNKNOWN IMPORTED)
      set_target_properties(Vorbis::vorbisenc PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${VORBISENC_INCLUDE_DIR}"
        IMPORTED_LOCATION "${VORBISENC_LIBRARY}"
        INTERFACE_LINK_LIBRARIES Vorbis::vorbis
      )
    endif ()
  endif ()

  mark_as_advanced(VORBISENC_INCLUDE_DIR VORBISENC_LIBRARY)

endif() # NOT TARGET Vorbis::vorbis OR NOT TARGET Vorbis::vorbisenc
