# Copyright 2025 Ingo Wald
# SPDX-License-Identifier: Apache-2.0

set(SOURCE_FILES
  faceIteration.h
  faceIteration.cpp
)

add_library(faceIteration STATIC ${SOURCE_FILES})
target_link_libraries(faceIteration owl::owl)
target_include_directories(faceIteration INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
)




