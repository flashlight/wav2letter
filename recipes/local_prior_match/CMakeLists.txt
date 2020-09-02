cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-recipes-models-local_prior_match)

# Build a version of the wav2letter++ object containing default wav2letter++ interfaces
# minus interface libraries which were custom defined, which are linked explicitly
add_library(
  wav2letter++_lpm_oss
  ""
  )

# Add custom targets
add_subdirectory(${PROJECT_SOURCE_DIR}/src/module) # for target module_lpm_oss
add_subdirectory(${PROJECT_SOURCE_DIR}/src/runtime) # for target runtime_lpm_oss

target_link_libraries(
  wav2letter++_lpm_oss
  PUBLIC
  wav2letter++
  runtime_lpm_oss
  module_lpm_oss
  )

# ------- Local Prior Matching (Off-Policy Proposal Model) -------
add_executable(
  Train_lpm_oss
  Train_lpm.cpp
)

target_include_directories(
  Train_lpm_oss
  PUBLIC
  ${PROJECT_SOURCE_DIR}/../../..
  )

target_link_libraries(
  Train_lpm_oss
  wav2letter++_lpm_oss
  )

# ------- Decode length -----
add_executable(
  decode_len_lpm
  Decode_length_lpm.cpp
)

target_link_libraries(
  decode_len_lpm
  PUBLIC
  wav2letter++
  )
