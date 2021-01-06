cmake_minimum_required(VERSION 3.10.0)

include(ExternalProject)

set(gtest_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/googletest/src/googletest/googletest/include)
set(gtest_URL https://github.com/google/googletest.git)
set(gtest_BUILD ${CMAKE_CURRENT_BINARY_DIR}/googletest/)
set(gtest_TAG 703bd9caab50b139428cea1aaff9974ebee5742e) # release 1.10.0

if (NOT TARGET gtest)
  # Download googletest
  ExternalProject_Add(
    gtest
    PREFIX googletest
    GIT_REPOSITORY ${gtest_URL}
    GIT_TAG ${gtest_TAG}
    BUILD_IN_SOURCE 1
    BUILD_COMMAND ${CMAKE_COMMAND} --build .
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
      -DBUILD_SHARED_LIBS:BOOL=${BUILD_SHARED_LIBS}
      -DCMAKE_BUILD_TYPE:STRING=Release
      -DBUILD_GMOCK:BOOL=ON
      -DBUILD_GTEST:BOOL=ON
      -Dgtest_force_shared_crt:BOOL=OFF
  )
endif ()

ExternalProject_Get_Property(gtest source_dir)
set(GTEST_SOURCE_DIR ${source_dir})
ExternalProject_Get_Property(gtest binary_dir)
set(GTEST_BINARY_DIR ${binary_dir})

if (BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
endif()

# Library and include dirs
set(GTEST_LIBRARIES
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gtest${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}"
)
set(GTEST_LIBRARIES_MAIN
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gtest_main${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}"
)
set(GMOCK_LIBRARIES
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gmock${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}"
)
set(GMOCK_LIBRARIES_MAIN
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_${LIB_TYPE}_LIBRARY_PREFIX}gmock_main${CMAKE_${LIB_TYPE}_LIBRARY_SUFFIX}"
)

set(GTEST_INCLUDE_DIRS ${GTEST_SOURCE_DIR}/googletest/include)
set(GMOCK_INCLUDE_DIRS ${GTEST_SOURCE_DIR}/googlemock/include)
# Make dirs so this can be used as an interface include directory
file(MAKE_DIRECTORY ${GTEST_INCLUDE_DIRS})
file(MAKE_DIRECTORY ${GMOCK_INCLUDE_DIRS})

add_library(GTest::gtest ${LIB_TYPE} IMPORTED)
add_library(GTest::gtest_main ${LIB_TYPE} IMPORTED)
add_library(GTest::gmock ${LIB_TYPE} IMPORTED)
add_library(GTest::gmock_main ${LIB_TYPE} IMPORTED)

set_target_properties(GTest::gtest PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIRS}
  IMPORTED_LOCATION ${GTEST_LIBRARIES}
  )
set_target_properties(GTest::gtest_main PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${GTEST_INCLUDE_DIRS}
  IMPORTED_LOCATION ${GTEST_LIBRARIES_MAIN}
  )
set_target_properties(GTest::gmock PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIRS}
  IMPORTED_LOCATION ${GMOCK_LIBRARIES}
  )
set_target_properties(GTest::gmock_main PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES ${GMOCK_INCLUDE_DIRS}
  IMPORTED_LOCATION ${GMOCK_LIBRARIES_MAIN}
  )
