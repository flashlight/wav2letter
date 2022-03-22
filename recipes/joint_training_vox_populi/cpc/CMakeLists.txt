cmake_minimum_required(VERSION 3.10)

set(common
  CPCCriterion.cpp
  CPCSpecAugment.cpp
  SequentialBuilder.cpp
  TransformerCPC.cpp
  MTLLoss.cpp
)

# ------------------------- Dedicated SSL Train executable --------------------------
add_executable(
  Train_cpc
  ${common}
  Train.cpp
)

target_link_libraries(
  Train_cpc
  flashlight::fl_pkg_speech
  flashlight::fl_pkg_runtime
)

add_executable(
  Decode_cpc
  ${common}
  Decode.cpp
)

target_link_libraries(
  Decode_cpc
  flashlight::fl_pkg_speech
  flashlight::fl_pkg_runtime
)

#add_executable(
#  Test_cpc
#  ${common}
#  Test.cpp
#  )

#target_link_libraries(
#  Test_cpc
#  flashlight::flashlight-app-asr
#)
