cmake_minimum_required(VERSION 3.5.1)

add_library(
  decoder-library
  INTERFACE
  )

target_sources(
  decoder-library
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/LexiconFreeDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Seq2SeqDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/TokenLMDecoder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Trie.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/WordLMDecoder.cpp
  )

target_link_libraries(
  decoder-library
  INTERFACE
  lm-library
  )
