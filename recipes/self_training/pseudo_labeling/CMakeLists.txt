cmake_minimum_required(VERSION 3.5.1)

add_executable(analyze_pseudo_label_dataset AnalyzeDataset.cpp)

find_package(flashlight REQUIRED)

target_link_libraries(
    analyze_pseudo_label_dataset
    PUBLIC
    flashlight::flashlight-app-asr
)

target_include_directories(
    analyze_pseudo_label_dataset
    PUBLIC
    ${PROJECT_SOURCE_DIR}
)
