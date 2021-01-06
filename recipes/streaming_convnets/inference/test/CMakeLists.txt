cmake_minimum_required(VERSION 3.10)

set(
    LIBS
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
    Threads::Threads
)

build_test(SRC Conv1dTest.cpp  LIBS ${LIBS})
build_test(SRC IdentityTest.cpp  LIBS ${LIBS})
build_test(SRC LayerNormTest.cpp  LIBS ${LIBS})
build_test(SRC LinearTest.cpp  LIBS ${LIBS})
build_test(SRC LogMelFeatureTest.cpp  LIBS ${LIBS})
build_test(SRC MemoryManagerTest.cpp  LIBS ${LIBS})
build_test(SRC ReluTest.cpp  LIBS ${LIBS})
build_test(SRC ResidualTest.cpp  LIBS ${LIBS})
build_test(SRC TDSBlockTest.cpp  LIBS ${LIBS})
