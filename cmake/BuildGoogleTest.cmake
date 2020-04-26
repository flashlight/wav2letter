cmake_minimum_required(VERSION 3.5.1)

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
    BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
    INSTALL_COMMAND ""
    CMAKE_CACHE_ARGS
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

# Library and include dirs
set(GTEST_LIBRARIES
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gtest_main${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gmock${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${GTEST_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}gmock_main${CMAKE_STATIC_LIBRARY_SUFFIX}"
)

set(GTEST_INCLUDE_DIR
  ${GTEST_SOURCE_DIR}/googletest/include
  ${GTEST_SOURCE_DIR}/googlemock/include
)
