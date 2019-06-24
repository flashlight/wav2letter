cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Dependencies -----------------------------
# Locate KenLM
find_package(kenlm REQUIRED)
if(kenlm_FOUND)
  message(STATUS "kenlm found")
else()
  message(FATAL_ERROR "kenlm not found")
endif()

# -------------------------- KenLM Dependencies --------------------------
# Issues with the kenlm build mean that some dependencies
# aren't properly linked on some systems - locate KenLM deps
set(KENLM_DEP_LIBRARIES "")
set(KENLM_DEP_INCLUDE_DIRS "")

# LZMA
find_package(LibLZMA)
if(LIBLZMA_FOUND)
  message(STATUS "LZMA found (library: ${LIBLZMA_LIBRARIES} include: ${LIBLZMA_INCLUDE_DIRS})")
  set(KENLM_DEP_LIBRARIES ${KENLM_DEP_LIBRARIES} ${LIBLZMA_LIBRARIES})
  set(KENLM_DEP_INCLUDE_DIRS ${KENLM_DEP_INCLUDE_DIRS} ${LIBLZMA_INCLUDE_DIRS})
else()
  message(STATUS "LZMA not found")
endif()

# BZip2
find_package(BZip2)
if(BZIP2_FOUND)
  message(STATUS "BZip2 found (library: ${BZIP2_LIBRARIES} include: ${BZIP2_INCLUDE_DIR})")
  set(KENLM_DEP_LIBRARIES ${KENLM_DEP_LIBRARIES} ${BZIP2_LIBRARIES})
  set(KENLM_DEP_INCLUDE_DIRS ${KENLM_DEP_INCLUDE_DIRS} ${BZIP2_INCLUDE_DIR})
else()
  message(STATUS "BZip2 not found")
endif()

# Z
find_package(ZLIB)
if(ZLIB_FOUND)
  message(STATUS "Z found (library: ${ZLIB_LIBRARIES} include: ${ZLIB_INCLUDE_DIR})")
  set(KENLM_DEP_LIBRARIES ${KENLM_DEP_LIBRARIES} ${ZLIB_LIBRARIES})
  set(KENLM_DEP_INCLUDE_DIRS ${KENLM_DEP_INCLUDE_DIRS} ${ZLIB_INCLUDE_DIR})
else()
  message(STATUS "Z not found")
endif()

if (NOT ZLIB_FOUND AND NOT BZIP2_FOUND AND NOT LIBLZMA_FOUND)
  message(FATAL_ERROR "Could not find a usable compression lib for KenLM (LZMA, BZ2, or Z)")
endif ()

# ----------------------------- Lib -----------------------------
add_library(
  lm
  INTERFACE
  )

target_sources(
  lm
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/KenLM.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConvLM.cpp
  )

target_link_libraries(
  lm
  INTERFACE
  flashlight::flashlight
  runtime
  decoder
  ${KENLM_LIBRARIES}
  ${KENLM_DEP_LIBRARIES}
  )

target_include_directories(
  lm
  INTERFACE
  ${KENLM_INCLUDE_DIRS}
  ${KENLM_DEP_INCLUDE_DIRS}
  )

# We need to specify options for building kenlm headers. These
# can be overrode in the build configuration
set(KENLM_MAX_ORDER 6 CACHE STRING "KENLM_MAX_ORDER")
target_compile_definitions(lm INTERFACE -DBMR_USE_QUICKSELECT)
target_compile_definitions(lm INTERFACE -DKENLM_MAX_ORDER=${KENLM_MAX_ORDER})
