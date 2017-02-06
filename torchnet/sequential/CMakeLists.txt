cmake_minimum_required (VERSION 2.8)
cmake_policy(VERSION 2.8)

set(PKGNAME torchnet/sequential)

file(GLOB_RECURSE luafiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.lua")

foreach(file ${luafiles})
  get_filename_component(dir ${file} PATH)
  install(FILES ${file} DESTINATION ${LUA_PATH}/${PKGNAME}/${dir})
endforeach()
