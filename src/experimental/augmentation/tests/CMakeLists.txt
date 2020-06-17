cmake_minimum_required(VERSION 3.5.1)

# Download and build googletest
include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake)

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  add_dependencies(${target} gtest) # make sure gtest is built first
  message("GTEST_LIBRARIES=${GTEST_LIBRARIES}")
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    ${GTEST_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    ${GTEST_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${target}
    PRIVATE
    "DATA_TEST_DATADIR=\"${PROJECT_SOURCE_DIR}/src/data/test/testdata\""
    "DECODER_TEST_DATADIR=\"${PROJECT_SOURCE_DIR}/src/decoder/test\""
    "FEATURE_TEST_DATADIR=\"${PROJECT_SOURCE_DIR}/src/libraries/feature/test/data\""
    "MODULE_TEST_ARCHDIR=\"${PROJECT_SOURCE_DIR}/src/module/test\""
    )
  add_test(${target} ${target})
endfunction(build_test)

if (W2L_BUILD_TESTS)
  # Augmentation
  build_test(${PROJECT_SOURCE_DIR}/src/experimental/augmentation/tests/AdditveNoiseTest.cpp)
  build_test(${PROJECT_SOURCE_DIR}/src/experimental/augmentation/tests/AudioAugmenterTest.cpp)

endif ()
