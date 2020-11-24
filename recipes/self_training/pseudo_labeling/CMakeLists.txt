cmake_minimum_required(VERSION 3.10)

add_executable(analyze_pseudo_label_dataset AnalyzeDataset.cpp)

target_link_libraries(
    analyze_pseudo_label_dataset
    PRIVATE
    flashlight::flashlight-app-asr
)

target_include_directories(
    analyze_pseudo_label_dataset
    PRIVATE
    ${PROJECT_SOURCE_DIR}
)
