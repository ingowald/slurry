
embed_ptx(
  OUTPUT_TARGET      devCode_ptx
  PTX_LINK_LIBRARIES faceIteration owl::owl
  SOURCES            devCode.cu
)

add_executable(miniApp hostCode.cpp)
target_link_libraries(miniApp PUBLIC
  devCode_ptx
  faceIteration
  rankCompositing
  )
