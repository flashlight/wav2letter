cmake_minimum_required(VERSION 3.10)

# Scripts which are common for our recipes
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/utilities/convlm_serializer)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/self_training/pseudo_labeling)

# TODO: fix
# add_subdirectory(${PROJECT_SOURCE_DIR}/local_prior_match)
