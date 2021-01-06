# Find gmock
#
#  GMOCK_INCLUDE_DIRS - where to find gmock/gmock.h, etc.
#  GMOCK_LIBRARIES   - List of libraries when using gmock.
#  GMOCK_FOUND       - True if gmock found.

if (GMOCK_INCLUDE_DIRS)
  # Already in cache, be silent
  set(GMOCK_FIND_QUIETLY TRUE)
endif()

find_package(GMock CONFIG)
if (NOT TARGET GTest::gmock)
  if (NOT GMOCK_ROOT)
    set(GMOCK_ROOT ENV{GMOCK_ROOT})
  endif()

  find_path(GMOCK_INCLUDE_DIRS gmock/gmock.h PATHS ${GMOCK_ROOT})
  find_library(GMOCK_MAIN_LIBRARY NAMES gmock_main PATHS ${GMOCK_ROOT})
  find_library(GMOCK_LIBRARIES NAMES gmock PATHS ${GMOCK_ROOT})

  set(GMOCK_BOTH_LIBRARIES
    ${GMOCK_MAIN_LIBRARY}
    ${GMOCK_LIBRARIES}
    )

  # handle the QUIETLY and REQUIRED arguments and set GMOCK_FOUND to TRUE if
  # all listed variables are TRUE
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(
    GMock
    DEFAULT_MSG
    GMOCK_MAIN_LIBRARY
    GMOCK_LIBRARIES
    GMOCK_LIBRARIES
    GMOCK_INCLUDE_DIRS
    )

  mark_as_advanced(
    GMOCK_MAIN_LIBRARY
    GMOCK_LIBRARIES
    LIBGTEST_LIBRARY
    GMOCK_LIBRARIES
    GMOCK_INCLUDE_DIRS
    )

  add_library(GTest::gmock UNKNOWN IMPORTED)
  set_target_properties(GTest::gmock PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIRS}
    IMPORTED_LOCATION ${GMOCK_LIBRARIES}
    )

  add_library(GTest::gmock_main UNKNOWN IMPORTED)
  set_target_properties(GTest::gmock_main PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIRS}
    IMPORTED_LOCATION ${GMOCK_MAIN_LIBRARY}
    )
endif()
