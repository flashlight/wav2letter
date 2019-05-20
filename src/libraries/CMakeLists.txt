cmake_minimum_required(VERSION 3.5.1)

# ------------------------- Components -------------------------

# common-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/common)

# criterion-library
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/criterion)

# ------------------------- Library -------------------------

add_library(
  wav2letter-libraries
  ""
  )

target_link_libraries(
  wav2letter-libraries
  PUBLIC
  common-library
  criterion-library
  )

target_include_directories(
  wav2letter-libraries
  PUBLIC
  ${PROJECT_SOURCE_DIR}/src
  )
