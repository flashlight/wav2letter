cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# common-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)

# criterion-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/criterion)

# decoder-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/decoder)

# feature-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/feature)

# lm-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lm)

# ------------------------- Library -------------------------

add_library(
  wav2letter-libraries
  ""
  )

set_target_properties(
  wav2letter-libraries
  PROPERTIES
  LINKER_LANGUAGE CXX
  CXX_STANDARD 11
  )

target_link_libraries(
  wav2letter-libraries
  PUBLIC
  common-library
  criterion-library
  decoder-library
  feature-library
  lm-library
  )

target_include_directories(
  wav2letter-libraries
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  )

target_compile_definitions(
  wav2letter-libraries
  PUBLIC
  ${W2L_COMPILE_DEFINITIONS}
  )
