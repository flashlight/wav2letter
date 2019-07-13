include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)

set_target_properties(
  wav2letter-libraries
  PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE
  TRUE
  )

function (add_pybind11_extension ext_name)
  pybind11_add_module(
    ${ext_name}
    ${CMAKE_CURRENT_SOURCE_DIR}/wav2letter/${ext_name}.cpp
    )

  target_link_libraries(
    ${ext_name}
    PRIVATE
    wav2letter-libraries
    )

  add_dependencies(${ext_name} pybind11)
endfunction ()

add_pybind11_extension(_common)
add_pybind11_extension(_criterion)
add_pybind11_extension(_decoder)
add_pybind11_extension(_feature)
