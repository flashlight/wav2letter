cmake_minimum_required(VERSION 3.5.1)

project(wav2letter++)

# Scripts which are common for our recipes
add_subdirectory(${PROJECT_SOURCE_DIR}/utilities/convlm_serializer)
#add_subdirectory(${PROJECT_SOURCE_DIR}/local_prior_match)
#add_subdirectory(${PROJECT_SOURCE_DIR}/self_training/pseudo_labeling)
