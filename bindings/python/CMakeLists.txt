include(${CMAKE_MODULE_PATH}/Buildpybind11.cmake)
include(${CMAKE_MODULE_PATH}/pybind11Tools.cmake)

set_target_properties(
  wav2letter-libraries
  PROPERTIES
  INTERFACE_POSITION_INDEPENDENT_CODE
  TRUE
  )

pybind11_add_module(
  _criterion
  ${CMAKE_CURRENT_SOURCE_DIR}/wav2letter/_criterion.cpp
  )

target_link_libraries(
  _criterion
  PRIVATE
  wav2letter-libraries
  )

add_dependencies(_criterion pybind11)
