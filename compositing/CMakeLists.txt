# Copyright 2025 Ingo Wald
# SPDX-License-Identifier: Apache-2.0

find_package(MPI REQUIRED)

set(SOURCE_FILES
  compositing.h
  compositing.cu
)

add_library(compositing STATIC ${SOURCE_FILES})
target_link_libraries(compositing owl::owl MPI::MPI_CXX)
target_include_directories(compositing INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
)





