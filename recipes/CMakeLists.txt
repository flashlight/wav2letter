cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++-recipes)

# Scripts which are common for our recipes
add_subdirectory(${PROJECT_SOURCE_DIR}/models/utilities/convlm_serializer)
add_subdirectory(${PROJECT_SOURCE_DIR}/models/local_prior_match)
add_subdirectory(${PROJECT_SOURCE_DIR}/models/self_training/pseudo_labeling)
