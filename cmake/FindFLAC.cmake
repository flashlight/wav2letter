# - Find FLAC
# Find the native FLAC includes and libraries
#
# Sets the following imported targets if FLAC is found:
# FLAC::FLAC
#
# Sets the following legacy CMake variables:
#  FLAC_INCLUDE_DIRS - where to find FLAC headers.
#  FLAC_LIBRARIES    - List of libraries when using libFLAC.
#  FLAC_FOUND        - True if libFLAC found.
#  FLAC_DEFINITIONS  - FLAC compile definitons

if (FLAC_INCLUDE_DIR)
    # Already in cache, be silent
    set (FLAC_FIND_QUIETLY TRUE)
endif ()

find_package (Ogg QUIET)

find_package (PkgConfig QUIET)
pkg_check_modules(PC_FLAC QUIET flac)

set(FLAC_VERSION ${PC_FLAC_VERSION})

find_path (FLAC_INCLUDE_DIR FLAC/stream_decoder.h
	HINTS
		${PC_FLAC_INCLUDEDIR}
		${PC_FLAC_INCLUDE_DIRS}
		${FLAC_ROOT}
	)

# MSVC built libraries can name them *_static, which is good as it
# distinguishes import libraries from static libraries with the same extension.
find_library (FLAC_LIBRARY
	NAMES
		FLAC
		libFLAC
		libFLAC_dynamic
		libFLAC_static
	HINTS
		${PC_FLAC_LIBDIR}
		${PC_FLAC_LIBRARY_DIRS}
		${FLAC_ROOT}
	)

# Handle the QUIETLY and REQUIRED arguments and set FLAC_FOUND to TRUE if
# all listed variables are TRUE.
include (FindPackageHandleStandardArgs)
find_package_handle_standard_args (FLAC
	REQUIRED_VARS
		FLAC_LIBRARY
		FLAC_INCLUDE_DIR
		OGG_FOUND
	VERSION_VAR
        FLAC_VERSION
	)

if (FLAC_FOUND)
	set (FLAC_INCLUDE_DIRS ${FLAC_INCLUDE_DIR})
	set (FLAC_LIBRARIES ${FLAC_LIBRARY} ${OGG_LIBRARIES})
	if (WIN32)
		set (FLAC_LIBRARIES ${FLAC_LIBRARIES} wsock32)
		get_filename_component (FLAC_LIBRARY_FILENAME ${FLAC_LIBRARY} NAME_WE)
		if (FLAC_LIBRARY_FILENAME MATCHES "libFLAC_static")
			set (FLAC_DEFINITIONS -DFLAC__NO_DLL)
		endif ()
	endif ()
    if (NOT TARGET FLAC::FLAC)
		add_library(FLAC::FLAC UNKNOWN IMPORTED)
		set_target_properties(FLAC::FLAC PROPERTIES
			INTERFACE_INCLUDE_DIRECTORIES "${FLAC_INCLUDE_DIR}"
			IMPORTED_LOCATION "${FLAC_LIBRARY}"
			INTERFACE_LINK_LIBRARIES
				Ogg::ogg
				$<$<BOOL:${WIN32}>:wsock32>
			INTERFACE_COMPILE_DEFINITIONS ${FLAC_DEFINITIONS}
			)
	endif ()
endif ()

mark_as_advanced(FLAC_INCLUDE_DIR FLAC_LIBRARY)
