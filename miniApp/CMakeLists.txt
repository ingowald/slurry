# Copyright 2025 Ingo Wald
# SPDX-License-Identifier: Apache-2.0


embed_ptx(
  OUTPUT_TARGET      embedded_devCode
  PTX_LINK_LIBRARIES faceIteration owl::owl
  SOURCES            devCode.cu
)

add_executable(miniApp
  miniApp.h
  hostCode.cpp
)
target_link_libraries(miniApp PUBLIC
  faceIteration
  compositing
  embedded_devCode
  )
