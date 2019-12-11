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
