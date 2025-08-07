# Copyright 2025 Ingo Wald
# SPDX-License-Identifier: Apache-2.0

find_package(MPI REQUIRED)

set(SOURCE_FILES
  rankCompositing.h
  rankCompositing.cu
)

add_library(rankCompositing STATIC ${SOURCE_FILES})
target_link_libraries(rankCompositing owl::owl MPI::MPI_CXX)
target_include_directories(rankCompositing INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
)





