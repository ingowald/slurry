// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "compositing.h"
#include "miniApp.h"
#include <cuda_runtime.h>
# define STB_IMAGE_WRITE_IMPLEMENTATION 1
# define STB_IMAGE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

/* has to match the name used in the embed_ptx cmake macro used in CMakeFile */
extern "C" const char devCode_ptx[];

namespace miniApp {
  
  typedef compositing::Context<Fragment,FinalCompositingResult> CompositingContext;
  vec2i fbSize { 800, 800 };

  void setCamera(PerLaunchData &launchData)
  {
    launchData.camera.org_00 = vec3f(-1,-1,-1);
    launchData.camera.org_du = vec3f(2,0,0);
    launchData.camera.org_dv = vec3f(0,2,0);
    launchData.camera.dir    = vec3f(0,0,1);
  }

  __global__ void g_localCompositing(FinalCompositingResult *results,
                                     const Fragment *fragments_allRanksMyPixels,
                                     int numPixels,
                                     // only for debugging...
                                     int numRanks)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPixels) return;

    bool dbg = (tid == 333*800+17);
    
    const Fragment *myFragments = fragments_allRanksMyPixels + tid * numRanks;
    FinalCompositingResult *myResult = results+tid;
    vec3f color = 0.f;
    float opacity = 0.f;
    for (int depth=0;depth<numRanks;depth++) {
      auto c = myFragments[depth];
      // if (dbg)
      // printf("frag[%i] = %f %f %f depth %f\n",depth,c.color.x,c.color.y,c.color.z,c.depth);
      color   += (1.f-opacity)*myFragments[depth].opacity*myFragments[depth].color;
      opacity += (1.f-opacity)*myFragments[depth].opacity;
    }
    // myResult->color = myFragments[0].color;
    myResult->color = color;
  }
  

  void localCompositing(FinalCompositingResult *results,
                        const Fragment *fragments,
                        int numPixels,
                        int numRanks)
  {
    int bs = 128;
    int nb = divRoundUp(numPixels,bs);
    g_localCompositing<<<nb,bs>>>(results,fragments,numPixels,numRanks);
  }


  void createModel(std::vector<vec3f> &vertices,
                   std::vector<vec3i> &indices,
                   int thisRank, int numRanks)
  {
    size_t FNV_PRIME = 0x00000100000001b3ull;

    float rectSpacing = 2.f/numRanks;
    float rectSize = 1.f / numRanks;
    float rectOffset = -1.f + .25f*rectSize;
    
    float shiftPerDepth = .8f*rectSize / numRanks;

    auto addBox = [&](int x, int y, int z)
    {
      float x0 = rectOffset + x * rectSpacing + z * shiftPerDepth;
      float y0 = rectOffset + y * rectSpacing + z * shiftPerDepth;
      float x1 = x0 + rectSize; 
      float y1 = y0 + rectSize;

      int i0 = vertices.size();
      vertices.push_back(vec3f(x0,y0,z));
      vertices.push_back(vec3f(x0,y1,z));
      vertices.push_back(vec3f(x1,y0,z));
      vertices.push_back(vec3f(x1,y1,z));
      indices.push_back(vec3i(i0)+vec3i(0,1,3));
      indices.push_back(vec3i(i0)+vec3i(0,3,2));

      for (auto vtx : vertices) PRINT(vtx);
    };
    for (int z=0;z<numRanks;z++)
      for (int y=0;y<numRanks;y++)
        for (int x=0;x<numRanks;x++) {
          size_t hash = 0x12345;
          hash = hash * FNV_PRIME ^ (x+123);
          hash = hash * FNV_PRIME ^ (y+456);
          int owner = (z + hash) % numRanks;
          if (owner == thisRank)
            addBox(x,y,z);
        }
  }

  void setScene(MPI_Comm comm,      
                faceIteration::Context *fit)
  {
    PING;
    int rank, size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);
#if 0
    // this is where you'd set your scene geometry ....
#else
    UserMeshData umd;
    umd.meshColor = owl::common::randomColor(13+rank);
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;
    createModel(vertices,indices,rank,size);
#endif
    PRINT(vertices.size());
    PRINT(indices.size());
    fit->setMesh(0,
                 vertices.data(),vertices.size(),
                 indices.data(),indices.size(),
                 &umd);
    PING;
  }
}
using namespace miniApp;


__global__
void resultToPixel(uint32_t *pixels, FinalCompositingResult *result, int numPixels)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPixels) return;

  int ix = tid % 800;
  int iy = tid / 800;
  // if (tid % 567 == 0)
  //   printf("pixel %i %i got color %f %f %f\n",
  //          ix,iy,
  //          result[tid].color.x,
  //          result[tid].color.y,
  //          result[tid].color.z);
  // if (result[tid].color.x != ix || result[tid].color.y != iy)
  //   printf("pixel %i %i got color %f %f %f\n",
  //          ix,iy,
  //          result[tid].color.x,
  //          result[tid].color.y,
  //          result[tid].color.z);
  pixels[tid] = owl::make_rgba(result[tid].color);
}

int main(int ac, char **av)
{
  std::string outFileName = "slurry.png";
  
  MPI_Comm comm = MPI_COMM_WORLD;
  // =============================================================================
  // init GPU - probably need to do some cleverness to figure ouw
  // which GPU you want to use per rank. or rely on
  // CUDA_VISIBLE_DEVICES being set...
  // =============================================================================
  int gpuID = 0;
  cudaSetDevice(gpuID);
  cudaFree(0);

  // =============================================================================
  // nit MPI - do this after gpu init so mpi can pick up on gpu.
  // =============================================================================
  int required = MPI_THREAD_MULTIPLE;
  int provided = 0;
  MPI_Init_thread(&ac,&av,required,&provided);
  int rank;
  MPI_Comm_rank(comm,&rank);
    
  // =============================================================================
  // initialize out compositing context
  // =============================================================================
  CompositingContext *comp
    = new CompositingContext(comm,
                             localCompositing);
  Fragment *localFB = comp->resize(fbSize);

  // =============================================================================
  // specify the geometry
  // =============================================================================
  faceIteration::Context *fit
    = faceIteration::Context::init(gpuID,sizeof(UserMeshData),1,
                                   sizeof(PerLaunchData),
                                   devCode_ptx,
                                   "faceIterationCallback",
                                   "miniApp_perPixel");
  PING;
  setScene(comm,fit);
  fit->build();
    
  PING;
  // =============================================================================
  // set up a launch, and issue launch to render local frame buffer
  // =============================================================================
  PerLaunchData launchData;
  launchData.localFB = localFB;
  launchData.rank = rank;
  PING;
  setCamera(launchData);
  PING;
  fit->launch(fbSize,&launchData);
  PING;


  // =============================================================================
  // composite the local frame buffers
  // =============================================================================
  FinalCompositingResult *composited
    = comp->run();
  PING;

  if (comp->mpi.rank == 0) {
    printf("saving pic...\n");
    uint32_t *frame = 0;
    int numPixels = fbSize.x*fbSize.y;
    cudaMallocManaged((void **)&frame,numPixels*sizeof(uint32_t));
    resultToPixel<<<divRoundUp(numPixels,128),128>>>(frame,composited,numPixels);
    cudaStreamSynchronize(0);
    stbi_flip_vertically_on_write(true);
    stbi_write_png(outFileName.c_str(),fbSize.x,fbSize.y,4,
                   frame,fbSize.x*sizeof(uint32_t));
    cudaFree(frame);
  }

  // =============================================================================
  // and wind down in reverse order
  // =============================================================================
  delete fit;
  delete comp;
    
  MPI_Finalize();
  return 0;
}
  
