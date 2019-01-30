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
  set(
    DATA_TEST_DATADIR
    "\"${CMAKE_SOURCE_DIR}/src/data/test/testdata\""
    )
  target_compile_definitions(
    DataTest
    PRIVATE
    -DDATA_TEST_DATADIR=${DATA_TEST_DATADIR}
  )
  # Decoder
  build_test(${CMAKE_SOURCE_DIR}/src/decoder/test/DecoderTest.cpp)
  set(
    DECODER_TEST_DATADIR
    "\"${CMAKE_SOURCE_DIR}/src/decoder/test\""
  )
  target_compile_definitions(
    DecoderTest
    PRIVATE
    -DDECODER_TEST_DATADIR=${DECODER_TEST_DATADIR}
  )
  # Feature
  set(
    FEATURE_TEST_DATADIR
    "\"${CMAKE_SOURCE_DIR}/src/feature/test/data\""
  )
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/CeplifterTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DctTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DerivativesTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/DitherTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/MfccTest.cpp)
  target_compile_definitions(
    MfccTest
    PRIVATE
    -DFEATURE_TEST_DATADIR=${FEATURE_TEST_DATADIR}
  )

  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/PreEmphasisTest.cpp)

  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/SoundTest.cpp)
  target_compile_definitions(
    SoundTest
    PRIVATE
    -DFEATURE_TEST_DATADIR=${FEATURE_TEST_DATADIR}
  )

  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/SpeechUtilsTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/TriFilterbankTest.cpp)
  build_test(${CMAKE_SOURCE_DIR}/src/feature/test/WindowingTest.cpp)
  # Module
  build_test(${CMAKE_SOURCE_DIR}/src/module/test/W2lModuleTest.cpp)
  set(
    MODULE_TEST_ARCHDIR
    "\"${CMAKE_SOURCE_DIR}/src/module/test\""
    )
  target_compile_definitions(
    W2lModuleTest
    PRIVATE
    -DMODULE_TEST_ARCHDIR=${MODULE_TEST_ARCHDIR}
  )
  # Runtime
  build_test(${CMAKE_SOURCE_DIR}/src/runtime/test/RuntimeTest.cpp)
endif ()
