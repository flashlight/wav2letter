#Download and create a rule to build FG GEMM
# sets FBGEMM_INCLUDE_DIR and FBGEMM_LIBRARIES

include(ExternalProject)

set(fbgemm_URL https://github.com/pytorch/FBGEMM.git)
set(fbgemm_BUILD ${CMAKE_CURRENT_BINARY_DIR}/fbgemm/)
set(fbgemm_TAG 58c002d1593f32aa420ab56b5c344e60d3fb6d05)

# Download fbgemm
if (NOT TARGET fbgemm)
  message("ExternalProject_Add(fbgemm) downloading ...")
  ExternalProject_Add(
      fbgemm
      PREFIX fbgemm
      GIT_REPOSITORY ${fbgemm_URL}
      GIT_TAG ${fbgemm_TAG}
      GIT_SUBMODULES
      BUILD_IN_SOURCE 1
      BUILD_COMMAND ${CMAKE_COMMAND} --build . --config Release
      INSTALL_COMMAND ""
      CMAKE_CACHE_ARGS
          -DFBGEMM_BUILD_BENCHMARKS:BOOL=OFF
          -DFBGEMM_BUILD_TESTS:BOOL=OFF
  )
endif()

ExternalProject_Get_Property(fbgemm source_dir)
set(FBGEMM_SOURCE_DIR ${source_dir})
ExternalProject_Get_Property(fbgemm binary_dir)
set(FBGEMM_BINARY_DIR ${binary_dir})

# Library and include dirs
set(fbgemm_LIBRARIES
  "${FBGEMM_SOURCE_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}fbgemm${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${FBGEMM_SOURCE_DIR}/cpuinfo/${CMAKE_STATIC_LIBRARY_PREFIX}cpuinfo${CMAKE_STATIC_LIBRARY_SUFFIX}"
  "${FBGEMM_SOURCE_DIR}/cpuinfo/deps/clog/${CMAKE_STATIC_LIBRARY_PREFIX}clog${CMAKE_STATIC_LIBRARY_SUFFIX}"
)
set(fbgemm_INCLUDE_DIRS
  ${CMAKE_CURRENT_BINARY_DIR}/fbgemm/src/fbgemm/include
  ${FBGEMM_SOURCE_DIR}/include
  ${FBGEMM_SOURCE_DIR}/third_party/cpuinfo/include
)
