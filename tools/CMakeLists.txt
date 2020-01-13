cmake_minimum_required(VERSION 3.5.1)

function(build_tool SRCFILE)
  get_filename_component(src_name ${SRCFILE} NAME_WE)
  set(target "${src_name}")
  add_executable(${target} ${SRCFILE})
  target_link_libraries(
    ${target}
    PRIVATE
    wav2letter++
    )
  target_include_directories(
    ${target}
    PRIVATE
    ${PROJECT_SOURCE_DIR}/..
    )
endfunction(build_tool)

if (W2L_BUILD_TOOLS)
   build_tool(${PROJECT_SOURCE_DIR}/tools/VoiceActivityDetection-CTC.cpp)
endif ()


# ---------------------- tds_module_converter ----------------------

add_executable(
  tds_module_converter
  ${CMAKE_CURRENT_LIST_DIR}/TDSModelConverter.cpp
)

add_dependencies(tds_module_converter common)
add_dependencies(tds_module_converter criterion)
add_dependencies(tds_module_converter module)
add_dependencies(tds_module_converter runtime)
add_dependencies(tds_module_converter GFLAGS)
add_dependencies(tds_module_converter streaming_inference_common)
add_dependencies(tds_module_converter streaming_inference_decoder)
add_dependencies(tds_module_converter streaming_inference_modules)

target_include_directories(
  tds_module_converter
  PRIVATE
    ${GFLAGS_INCLUDE_DIRS}
)

target_link_libraries(
  tds_module_converter
  PRIVATE
    common
    criterion
    module
    runtime
    ${GFLAGS_LIBRARIES}
    streaming_inference_common
    streaming_inference_decoder
    streaming_inference_modules
)
