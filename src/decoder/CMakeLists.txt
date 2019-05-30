cmake_minimum_required(VERSION 3.5.1)

# ----------------------------- Lib -----------------------------
add_library(
  decoder
  INTERFACE
  )

target_sources(
  decoder
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TokenLMDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Trie.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/WordLMDecoder.cpp
  )

target_link_libraries(
  decoder
  INTERFACE
  ${GLOG_LIBRARIES}
  )

target_include_directories(
  decoder
  INTERFACE
  ${GLOG_INCLUDE_DIRS}
  )
