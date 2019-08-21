cmake_minimum_required(VERSION 3.5.1)

add_library(
  lm-library
  INTERFACE
  )

# ------------------------- KenLM-specific -------------------------

if (W2L_LIBRARIES_USE_KENLM)
  find_package(kenlm REQUIRED)

  # Required for KenLM to read ARPA files in compressed format
  find_package(LibLZMA)
  find_package(BZip2)
  find_package(ZLIB)

  target_sources(
    lm-library
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/KenLM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ConvLM.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ZeroLM.cpp
    )

  target_link_libraries(
    lm-library
    INTERFACE
    ${KENLM_LIBRARIES}
    ${LIBLZMA_LIBRARIES}
    ${BZIP2_LIBRARIES}
    ${ZLIB_LIBRARIES}
    )

  target_include_directories(
    lm-library
    INTERFACE
    ${KENLM_INCLUDE_DIRS}
    ${LIBLZMA_INCLUDE_DIRS}
    ${BZIP2_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
    )

  target_compile_definitions(
    lm-library
    INTERFACE
    KENLM_MAX_ORDER=${KENLM_MAX_ORDER}
    )
endif ()
