cmake_minimum_required(VERSION 3.5.1)

# Main library
add_library(
  criterion
  INTERFACE
  )

# ---------------------------- Backend-specific -----------------------------

if (FLASHLIGHT_USE_CUDA)
  # Build WarpCTC
  set(WARPCTC_DIR ${PROJECT_SOURCE_DIR}/src/third_party/warpctc)
  message(STATUS "Adding warpctc:")
  add_subdirectory(${WARPCTC_DIR} ${PROJECT_BINARY_DIR}/src/third_party/warpctc)
  set(WARPCTC_INCLUDE_DIR ${WARPCTC_DIR}/include)

  target_sources(
    criterion
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/CriterionUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cuda/FullConnectionCriterion.cpp
    )

  target_link_libraries(
    criterion
    INTERFACE
    ${CUDA_LIBRARIES}
    warpctc # added directly from subdir
    # Can remove once CTC constrained Viterbi is implemented in CUDA T62268085
    criterion-library
    )

  target_include_directories(
    criterion
    INTERFACE
    ${CUDA_INCLUDE_DIRS}
    ${WARPCTC_INCLUDE_DIR}
    )
elseif (FLASHLIGHT_USE_CPU)
  target_sources(
    criterion
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/ConnectionistTemporalClassificationCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/CriterionUtils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/ForceAlignmentCriterion.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/backend/cpu/FullConnectionCriterion.cpp
    )
else ()
  message(
    FATAL_ERROR
    "wav2letter criterions not implemented for ${FLASHLIGHT_BACKEND} backend"
    )
endif ()

# ----------------------------- Lib -----------------------------

target_sources(
  criterion
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ConnectionistTemporalClassificationCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/CriterionUtils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ForceAlignmentCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Seq2SeqCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/FullConnectionCriterion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TransformerCriterion.cpp
  )

target_link_libraries(
  criterion
  INTERFACE
  attention
  window
  common
  wav2letter-libraries
  flashlight::flashlight
  ${GLOG_LIBRARIES}
  ${cereal_LIBRARIES}
  )

target_include_directories(
  criterion
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  ${cereal_INCLUDE_DIRS}
  )

# ---------------------------- Attention/Window -----------------------------
# Attention
add_library(
  attention
  INTERFACE
  )

target_sources(
  attention
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/ContentAttention.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/LocationAttention.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/MultiHeadAttention.cpp
  )

target_include_directories(
  attention
  INTERFACE
  attention
  src/flashlight
  ${cereal_INCLUDE_DIRS}
  )

target_link_libraries(
  attention
  INTERFACE
  flashlight::flashlight
  common
  ${cereal_LIBRARIES}
  )

# Window
add_library(
  window
  INTERFACE
  )

target_sources(
  window
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/MedianWindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/SoftPretrainWindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/SoftWindow.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/attention/StepWindow.cpp
  )

target_include_directories(
  window
  INTERFACE
  attention
  src/flashlight
  ${cereal_INCLUDE_DIRS}
  )

target_link_libraries(
  window
  INTERFACE
  flashlight::flashlight
  common
  ${cereal_LIBRARIES}
  )
