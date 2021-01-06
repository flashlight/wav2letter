# Try to find Cereal
#
# Sets the following imported targets if cereal is found with a config:
# cereal
#
# If cereal is not found with a CMake config, legacy variables are set:
# cereal_FOUND
# cereal_INCLUDE_DIRS - directories with Cereal headers
# cereal_DEFINITIONS - Cereal compiler flags

find_package(cereal CONFIG)

if (NOT TARGET cereal)
  find_path(cereal_header_paths_tmp
    NAMES
    cereal.hpp
    PATH_SUFFIXES
    include
    cereal/include
	  PATHS
    ${CEREAL_ROOT_DIR}
    ${CEREAL_ROOT_DIR}/include
    ${CEREAL_ROOT_DIR}/cereal/include
    $ENV{CEREAL_ROOT_DIR}
    $ENV{CEREAL_ROOT_DIR}/include
    $ENV{CEREAL_ROOT_DIR}/cereal
    )

  get_filename_component(cereal_INCLUDE_DIRS ${cereal_header_paths_tmp} PATH)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(cereal
    REQUIRED_VARS cereal_INCLUDE_DIRS
    )

  message(STATUS "Found cereal (include: ${cereal_INCLUDE_DIRS})")
  mark_as_advanced(cereal_FOUND)
  if (cereal_FOUND)
    add_library(cereal INTERFACE IMPORTED)
    set_target_properties(cereal PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${cereal_INCLUDE_DIRS}")
  endif()
endif()
