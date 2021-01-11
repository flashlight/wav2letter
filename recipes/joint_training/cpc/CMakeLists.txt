cmake_minimum_required(VERSION 3.10)

#project(wav2letter++-experimental-cpc)


#find_package(flashlight REQUIRED)

# ------------------------- Dedicated SSL Train executable --------------------------
add_executable(
  Train_cpc
  CPCCriterion.cpp
  Train.cpp
  )

target_link_libraries(
  Train_cpc
  flashlight::flashlight-app-asr
)

add_executable(
  Decode_cpc
  CPCCriterion.cpp
  Decode.cpp
  )

target_link_libraries(
  Decode_cpc
  flashlight::flashlight-app-asr
)

add_executable(
  Test_cpc
  CPCCriterion.cpp
  Test.cpp
  )

target_link_libraries(
  Test_cpc
  flashlight::flashlight-app-asr
)
