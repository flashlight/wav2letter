# Try to find Cereal
#
# Sets the following variables:
# CEREAL_FOUND
# CEREAL_INCLUDE_DIRS - directories with Cereal headers
# CEREAL_DEFINITIONS - Cereal compiler flags

find_path(CEREAL_INCLUDE_DIR
  cereal
	HINTS
    "$ENV{CEREAL_ROOT}/include"
    "/usr/include"
    "$ENV{PROGRAMFILES}/cereal/include"
)

set(CEREAL_INCLUDE_DIRS ${CEREAL_INCLUDE_DIR})

# sets cereal_FOUND value based on validity of CEREAL_INCLUDE_DIRS
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(cereal DEFAULT_MSG CEREAL_INCLUDE_DIRS)
