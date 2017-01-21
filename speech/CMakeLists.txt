cmake_minimum_required (VERSION 2.8)

set(CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake"
  "${CMAKE_MODULE_PATH}")

find_package(Torch REQUIRED)
find_package(FFTW REQUIRED)

set(src
  init.c
)

set(luasrc
  ceplifter.lua
  dct.lua
  derivatives.lua
  fftw.lua
  init.lua
  mfcc.lua
  mfsc.lua
  pow.lua
  preemphasis.lua
  proc.lua
  trifiltering.lua
  windowing.lua
)

add_torch_package(speech "${src}" "${luasrc}")
# target_link_libraries(beamer ${kenlmlib})
