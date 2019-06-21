# Using FindCBLAS.cmake from the following repo
# https://github.com/clementfarabet/THC/blob/master/COPYRIGHT.txt

# - Find CBLAS library
#
# This module finds an installed fortran library that implements the CBLAS
# linear-algebra interface (see http://www.netlib.org/blas/), with CBLAS
# interface.
#
# This module sets the following variables:
#  CBLAS_FOUND - set to true if a library implementing the CBLAS interface is found
#  CBLAS_LIBRARIES - list of libraries (using full path name) to link against to use CBLAS
#  CBLAS_INCLUDE_DIR - path to includes
#  CBLAS_INCLUDE_FILE - the file to be included to use CBLAS
#  MKL_BLAS_FOUND - set if MKL is found

SET(CBLAS_LIBRARIES CACHE STRING
  "Path to CBLAS Library")
SET(CBLAS_INCLUDE_DIR CACHE STRING
  "Path to CBLAS include directory")
SET(CBLAS_INCLUDE_FILE CACHE STRING
  "CBLAS header name")


# If a valid PkgConfig configuration for cblas is found, this overrides and cancels
# all further checks.
FIND_PACKAGE(PkgConfig)
IF(PKG_CONFIG_FOUND)
  PKG_CHECK_MODULES(PC_CBLAS cblas)
ENDIF(PKG_CONFIG_FOUND)

IF(PC_CBLAS_FOUND)

  FOREACH(PC_LIB ${PC_CBLAS_LIBRARIES})
    FIND_LIBRARY(${PC_LIB}_LIBRARY NAMES ${PC_LIB} HINTS ${PC_CBLAS_LIBRARY_DIRS} )
    IF (NOT ${PC_LIB}_LIBRARY)
      message(FATAL_ERROR "Something is wrong in your pkg-config file - lib ${PC_LIB} not found in ${PC_CBLAS_LIBRARY_DIRS}")
    ENDIF (NOT ${PC_LIB}_LIBRARY)
    LIST(APPEND CBLAS_LIBRARIES ${${PC_LIB}_LIBRARY})
  ENDFOREACH(PC_LIB)

  FIND_PATH(CBLAS_INCLUDE_DIRS NAMES cblas.h HINTS ${PC_CBLAS_INCLUDE_DIRS} )
  IF (NOT CBLAS_INCLUDE_DIRS)
    message(FATAL_ERROR "Something is wrong in your pkg-config file - cblas.h not found in ${PC_CBLAS_INCLUDE_DIRS}")
  ENDIF (NOT CBLAS_INCLUDE_DIRS)
  SET(CBLAS_INCLUDE_DIR ${CBLAS_INCLUDE_DIRS})

  FIND_PACKAGE_HANDLE_STANDARD_ARGS(CBLAS DEFAULT_MSG CBLAS_LIBRARIES CBLAS_INCLUDE_DIR)
  MARK_AS_ADVANCED(
    CBLAS_LIBRARIES
    CBLAS_INCLUDE_DIR
    CBLAS_INCLUDE_DIRS)

ELSE(PC_CBLAS_FOUND)

SET(INTEL_MKL_ROOT_DIR CACHE STRING
  "Root directory of the Intel MKL")

SET(CBLAS_ROOT_DIR CACHE STRING
  "Root directory for custom CBLAS implementation")

MARK_AS_ADVANCED(INTEL_MKL_ROOT_DIR CBLAS_ROOT_DIR)

INCLUDE(CheckTypeSize)
CHECK_TYPE_SIZE("void*" SIZE_OF_VOIDP)

IF (NOT INTEL_MKL_ROOT_DIR)
  SET(INTEL_MKL_ROOT_DIR $ENV{INTEL_MKL_ROOT})
ENDIF()

IF(NOT CBLAS_ROOT_DIR)

  IF (ENV{CBLASDIR})
    SET(CBLAS_ROOT_DIR $ENV{CBLASDIR})
    IF ("${SIZE_OF_VOIDP}" EQUAL 8)
        SET(CBLAS_LIB64_DIR "${CBLAS_ROOT_DIR}/lib64")
    ELSE()
        SET(CBLAS_LIB32_DIR "${CBLAS_ROOT_DIR}/lib")
    ENDIF()
  ENDIF()

  IF (ENV{CBLAS_ROOT_DIR})
    SET(CBLAS_ROOT_DIR $ENV{CBLAS_ROOT_DIR})
    IF ("${SIZE_OF_VOIDP}" EQUAL 8)
        SET(CBLAS_LIB64_DIR "${CBLAS_ROOT_DIR}/lib64")
    ELSE()
        SET(CBLAS_LIB32_DIR "${CBLAS_ROOT_DIR}/lib")
    ENDIF()
  ENDIF()

  IF (INTEL_MKL_ROOT_DIR)
    SET(CBLAS_ROOT_DIR ${INTEL_MKL_ROOT_DIR})
    IF(APPLE)
        IF ("${SIZE_OF_VOIDP}" EQUAL 8)
            SET(CBLAS_LIB64_DIR "${CBLAS_ROOT_DIR}/lib")
        ELSE()
            SET(CBLAS_LIB32_DIR "${CBLAS_ROOT_DIR}/lib")
        ENDIF()
    ELSE(APPLE) # Windows and Linux
        IF ("${SIZE_OF_VOIDP}" EQUAL 8)
            SET(CBLAS_LIB64_DIR "${CBLAS_ROOT_DIR}/lib/intel64")
        ELSE()
            SET(CBLAS_LIB32_DIR "${CBLAS_ROOT_DIR}/lib/ia32")
        ENDIF()
    ENDIF(APPLE)
  ENDIF()
ENDIF()

if(CBLAS_ROOT_DIR)
  set(CBLAS_INCLUDE_DIR "${CBLAS_ROOT_DIR}/include")
endif()

# Old CBLAS search
SET(_verbose TRUE)
INCLUDE(CheckFunctionExists)
INCLUDE(CheckIncludeFile)

MACRO(CHECK_ALL_LIBRARIES
    LIBRARIES
    _prefix
    _name
    _flags
    _list
    _include
    _search_include
    _libraries_work_check)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  SET(__list)
  FOREACH(_elem ${_list})
    IF(__list)
      SET(__list "${__list} - ${_elem}")
    ELSE(__list)
      SET(__list "${_elem}")
    ENDIF(__list)
  ENDFOREACH(_elem)
  IF(_verbose)
    MESSAGE(STATUS "Checking for [${__list}]")
  ENDIF(_verbose)
  SET(_libraries_work TRUE)
  SET(${LIBRARIES})
  SET(_combined_name)
  SET(_paths)
  FOREACH(_library ${_list})
    SET(_combined_name ${_combined_name}_${_library})
    # did we find all the libraries in the _list until now?
    # (we stop at the first unfound one)
    IF(_libraries_work)
      IF(APPLE)
        FIND_LIBRARY(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
          ENV DYLD_LIBRARY_PATH
          "${CBLAS_LIB_DIR}" "${CBLAS_LIB32_DIR}" "${CBLAS_LIB64_DIR}"
          )
      ELSE(APPLE)
        FIND_LIBRARY(${_prefix}_${_library}_LIBRARY
          NAMES ${_library} lib${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
          ENV LD_LIBRARY_PATH
          "${CBLAS_LIB_DIR}" "${CBLAS_LIB32_DIR}" "${CBLAS_LIB64_DIR}"
          PATH_SUFFIXES atlas
          )
        IF(NOT ${_prefix}_${library}_LIBRARY)
            LIST(APPEND CMAKE_FIND_LIBRARY_SUFFIXES ".so.3")
            FIND_LIBRARY(${_prefix}_${_library}_LIBRARY
              NAMES ${_library}
              PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64
              ENV LD_LIBRARY_PATH
              "${CBLAS_LIB_DIR}" "${CBLAS_LIB32_DIR}" "${CBLAS_LIB64_DIR}"
              PATH_SUFFIXES atlas
              )
          ENDIF(NOT ${_prefix}_${library}_LIBRARY)
      ENDIF(APPLE)
      MARK_AS_ADVANCED(${_prefix}_${_library}_LIBRARY)

      SET(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      SET(_libraries_work ${${_prefix}_${_library}_LIBRARY})
    ENDIF(_libraries_work)
  ENDFOREACH(_library)

  # Test include
  SET(_bug_search_include ${_search_include}) #CMAKE BUG!!! SHOULD NOT BE THAT
  SET(_bug_libraries_work_check ${_libraries_work_check}) #CMAKE BUG!!! SHOULD NOT BE THAT

  IF(_bug_search_include)
    FIND_PATH(${_prefix}${_combined_name}_INCLUDE ${_include}
      PATHS
      ${CBLAS_ROOT_DIR}/include
      /opt/intel/mkl/include
      /usr/include
      /usr/local/include
      /sw/include
      /opt/local/include
      PATH_SUFFIXES
      openblas
      )
    MARK_AS_ADVANCED(${_prefix}${_combined_name}_INCLUDE)

    IF(${_prefix}${_combined_name}_INCLUDE)
      IF (_verbose)
        MESSAGE(STATUS "Includes found")
      ENDIF (_verbose)
      SET(${_prefix}_INCLUDE_DIR ${${_prefix}${_combined_name}_INCLUDE})
      SET(${_prefix}_INCLUDE_FILE ${_include})
    ELSE(${_prefix}${_combined_name}_INCLUDE)
      SET(_libraries_work FALSE)
    ENDIF(${_prefix}${_combined_name}_INCLUDE)

  ELSE(_bug_search_include)
    SET(${_prefix}_INCLUDE_DIR)
    SET(${_prefix}_INCLUDE_FILE ${_include})
  ENDIF(_bug_search_include)

  IF (_bug_libraries_work_check)
    # Test this combination of libraries.
    IF(_libraries_work)
      SET(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
      CHECK_FUNCTION_EXISTS(${_name} ${_prefix}${_combined_name}_WORKS)
      SET(CMAKE_REQUIRED_LIBRARIES)
      MARK_AS_ADVANCED(${_prefix}${_combined_name}_WORKS)
      SET(_libraries_work ${${_prefix}${_combined_name}_WORKS})

      IF(_verbose AND _libraries_work)
        MESSAGE(STATUS "CBLAS Symbols FOUND")
      ELSE()
        MESSAGE(STATUS "CBLAS Symbols NOTFOUND")
      ENDIF(_verbose AND _libraries_work)

    ENDIF(_libraries_work)
  ENDIF()

  # Fin
  IF(NOT _libraries_work)
    SET(${LIBRARIES} NOTFOUND)
  ENDIF(NOT _libraries_work)
ENDMACRO(CHECK_ALL_LIBRARIES)

# Apple CBLAS library?
IF(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "Accelerate"
    "Accelerate/Accelerate.h"
    FALSE,
    TRUE)
ENDIF(NOT CBLAS_LIBRARIES)

IF( NOT CBLAS_LIBRARIES )
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "vecLib"
    "vecLib/vecLib.h"
    FALSE,
    TRUE)
ENDIF( NOT CBLAS_LIBRARIES )

# CBLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
IF(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "cblas;atlas"
    "cblas.h"
    TRUE,
    TRUE)
ENDIF(NOT CBLAS_LIBRARIES)

# OpenBLAS library
IF(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "openblas"
    "cblas.h"
    TRUE,
    TRUE)
ENDIF(NOT CBLAS_LIBRARIES)

# Generic CBLAS library
IF(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    cblas_dgemm
    ""
    "cblas"
    "cblas.h"
    TRUE,
    TRUE)
ENDIF(NOT CBLAS_LIBRARIES)

# Generic BLAS+CBLAS library
# Debian based systems have them as single library
IF(NOT CBLAS_LIBRARIES)
  CHECK_ALL_LIBRARIES(
    CBLAS_LIBRARIES
    CBLAS
    dgemm_
    ""
    "blas"
    "cblas.h"
    TRUE,
    TRUE)
ENDIF(NOT CBLAS_LIBRARIES)

IF(CBLAS_LIBRARIES)
  IF (NOT MKL_CBLAS_FOUND)
    SET(CBLAS_FOUND TRUE)
  ENDIF()
ELSE(CBLAS_LIBRARIES)
  SET(CBLAS_FOUND FALSE)
ENDIF(CBLAS_LIBRARIES)

IF(NOT CBLAS_FOUND AND CBLAS_FIND_REQUIRED)
  IF(NOT MKL_CBLAS_FOUND)
    MESSAGE(FATAL_ERROR "CBLAS library not found. Please specify library location")
  ENDIF()
ENDIF(NOT CBLAS_FOUND AND CBLAS_FIND_REQUIRED)
IF(NOT CBLAS_FIND_QUIETLY)
  IF(CBLAS_FOUND OR MKL_CBLAS_FOUND)
    MESSAGE(STATUS "CBLAS library found")
  ELSE()
    MESSAGE(STATUS "CBLAS library not found.")
  ENDIF()
ENDIF(NOT CBLAS_FIND_QUIETLY)

ENDIF(PC_CBLAS_FOUND)

MARK_AS_ADVANCED(
    CBLAS_INCLUDE_DIR
    CBLAS_INCLUDE_FILE
    CBLAS_LIBRARIES
    cblas_LIBRARY
    blas_LIBRARY)
