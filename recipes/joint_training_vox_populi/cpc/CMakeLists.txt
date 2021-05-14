cmake_minimum_required(VERSION 3.10)

#project(wav2letter++-experimental-cpc)


#find_package(flashlight REQUIRED)
set(common
   CPCCriterion.cpp
   CPCSpecAugment.cpp
   SequentialBuilder.cpp
   TransformerCPC.cpp)

# ------------------------- Dedicated SSL Train executable --------------------------
add_executable(
  Train_cpc
  ${common}
  Train.cpp
  )

target_link_libraries(
  Train_cpc
  flashlight::flashlight-app-asr
  flashlight::flashlight-app-common
)

add_executable(
  Decode_cpc
  ${common}
  Decode.cpp
)

target_link_libraries(
  Decode_cpc
  flashlight::flashlight-app-asr
  flashlight::flashlight-app-common
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
