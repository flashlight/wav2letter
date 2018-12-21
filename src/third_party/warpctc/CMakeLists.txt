IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "warpctc: cuda found ${CUDA_FOUND}")

option(WITH_GPU "compile warp-ctc with cuda." ${CUDA_FOUND})
option(WITH_OMP "compile warp-ctc with openmp." ON)

if(NOT WITH_OMP)
    add_definitions(-DCTC_DISABLE_OMP)
endif()

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

IF (CUDA_VERSION GREATER 7.6)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_61,code=sm_61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_62,code=sm_62")
ENDIF()

IF ((CUDA_VERSION VERSION_GREATER "9.0") OR (CUDA_VERSION VERSION_EQUAL "9.0"))
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_70,code=sm_70")
    message(STATUS "warpctc: using CUDA 9.0 or above")
ENDIF()

if (NOT APPLE)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")
ENDIF()

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "warpctc: DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


IF (WITH_GPU)

    MESSAGE(STATUS "warpctc: Building shared library with GPU support")

    CUDA_ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cu src/reduce.cu)
    
    TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})
    

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

ELSE()
    MESSAGE(STATUS "warpctc: Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")

ENDIF()


