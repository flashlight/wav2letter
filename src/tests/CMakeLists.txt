cmake_minimum_required(VERSION 3.5.1)

# Download and build googletest
include(${CMAKE_MODULE_PATH}/BuildGoogleTest.cmake)

function(build_test SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  add_dependencies(${target} gtest) # make sure gtest is built first
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    ${GTEST_LIBRARIES}
    ${GMOCK_LIBRARIES}
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${CMAKE_SOURCE_DIR}/..
    ${GTEST_INCLUDE_DIR}
    ${GMOCK_INCLUDE_DIR}
    )
  target_compile_definitions(
    ${target}
    PRIVATE
    "DATA_TEST_DATADIR=\"${CMAKE_SOURCE_DIR}/src/data/test/testdata\""
    "DECODER_TEST_DATADIR=\"${CMAKE_SOURCE_DIR}/src/decoder/test\""
    "FEATURE_TEST_DATADIR=\"${CMAKE_SOURCE_DIR}/src/feature/test/data\""
    "MODULE_TEST_ARCHDIR=\"${CMAKE_SOURCE_DIR}/src/module/test\""
    )
  add_test(${target} ${target})
endfunction(build_test)

if (W2L_BUILD_TESTS)
  # Common
  build_test(${CMAKE_SOURCE_DIR}/src/common/test/W2lCommonTest.cpp)
  # Criterion
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/test/CriterionTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/test/Seq2SeqTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/attention/test/AttentionTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/criterion/attention/test/WindowTest.cpp)
  # Data
  build_test(${CMAKE_SOURCE_DIR}/src/data/test/DataTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/data/test/SoundTest.cpp)
  # Decoder
  build_test(${CMAKE_SOURCE_DIR}/src/decoder/test/DecoderTest.cpp)
  # Feature
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/CeplifterTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DctTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DerivativesTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DitherTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/MfccTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/PreEmphasisTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/SpeechUtilsTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/TriFilterbankTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/WindowingTest.cpp)
  # Module
  build_test(${CMAKE_SOURCE_DIR}/src/module/test/W2lModuleTest.cpp)
  # Runtime
  build_test(${CMAKE_SOURCE_DIR}/src/runtime/test/RuntimeTest.cpp)
endif ()
