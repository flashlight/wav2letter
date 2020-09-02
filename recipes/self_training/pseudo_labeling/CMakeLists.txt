cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-recipes-models-self_training-pseudo_labeling)

add_executable(analyze_pseudo_label_dataset AnalyzeDataset.cpp)

target_link_libraries(
    analyze_pseudo_label_dataset
    PUBLIC
    wav2letter++
)

target_include_directories(
    analyze_pseudo_label_dataset
    PUBLIC
    ${CMAKE_PROJECT_SOURCE}
)
