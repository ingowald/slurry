// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "compositing.h"
#include "miniApp.h"
#include <cuda_runtime.h>
# define STB_IMAGE_WRITE_IMPLEMENTATION 1
# define STB_IMAGE_IMPLEMENTATION 1
# include "stb/stb_image.h"
# include "stb/stb_image_write.h"

/* has to match the name of the file used in the embed_ptx cmake macro
   used in CMakeFile */
extern "C" const char devCode_ptx[];

namespace miniApp {

    struct Camera {
      // for this simple example, let's use a orthogonal camera
      vec3f org_00;
      vec3f org_du;
      vec3f org_dv;
      vec3f dir;
    };
  
  typedef compositing::Context<Fragment,FinalCompositingResult> CompositingContext;
  vec2i fbSize { 800, 800 };

  void setCamera(Camera &camera)
  {
    camera.org_00 = vec3f(-1,-1,-1);
    camera.org_du = vec3f(2,0,0);
    camera.org_dv = vec3f(0,2,0);
    camera.dir    = vec3f(0,0,1);
  }

  /*! do whatever kind of compositing the app wants to do - input is a
      list of fragments sorted by a) (major) pixel ID and b) (minor)
      depth. Each rank produces exactly one fragment per pixel, so the
      total number of fragments is `numPixelsOnThisRank*numRanks`,
      sorted such that the first `numRanks` fragments are those for
      the first pixel on this rank (sorted in increasing depth), then
      `numRanks` fragments for the second pixel (again sorted in
      depth), etc. Output is one final composited value (of type
      FinalCompositingResult, whatever that might be, per pixel on
      this rank */
  __global__
  void g_localCompositing(FinalCompositingResult *results,
                          const Fragment *fragments_allRanksMyPixels,
                          /*! number of pixels we are responsible for
                            compositing on this rank */
                          int numPixelsOnThisRank,
                          /*! number of fragments (one per rank) for
                              each pixel */
                          int numRanks)
  {
    /* IMPORTANT: though we do standard 'over'-compositing here, with
       colors, alpha, etcpp; but there is no implicit assumption
       anywhere in 'composite.h' that this is what is being done. You
       can change the 'FinalCompositingResult' and 'Fragment' type to
       hold whatever you want (as long as the fragment is derived from
       `slurry::Fragment`), and do whatever compositing you want */
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPixelsOnThisRank) return;

    const Fragment *myFragments = fragments_allRanksMyPixels + tid * numRanks;
    FinalCompositingResult *myResult = results+tid;
    vec3f color = 0.f;
    float opacity = 0.f;
    for (int depth=0;depth<numRanks;depth++) {
      color   += (1.f-opacity)*myFragments[depth].opacity*myFragments[depth].color;
      opacity += (1.f-opacity)*myFragments[depth].opacity;
    }
    myResult->color = color;
  }
  
  /*! a c-style function that launches the cuda kernel, we need a
      separate C linkage function to pass a function pointer to the
      compositing interface (which allows that compositing interface
      to not care what kidn of compositing the user wants to do */
  void localCompositing(FinalCompositingResult *results,
                        const Fragment *fragments,
                        int numPixelsOnThisRank,
                        int numRanks)
  {
    int bs = 128;
    int nb = divRoundUp(numPixelsOnThisRank,bs);
    g_localCompositing<<<nb,bs>>>(results,fragments,numPixelsOnThisRank,numRanks);
  }

  /*! in this mini app, we use this to create some kind of test model;
      what geometry you want to pass to the faceIteration interface
      can be changed to whatever you want */
  faceIteration::Context *setScene(MPI_Comm comm, int gpuID)
  {
    /* sample app creates one mesh per rank, with variour different
       depth-overlapping squares */
    struct {
      int rank, size;
    } mpi;
    MPI_Comm_rank(comm,&mpi.rank);
    MPI_Comm_size(comm,&mpi.size);
    
    std::vector<vec3i> indices;
    std::vector<vec3f> vertices;

    size_t FNV_PRIME = 0x00000100000001b3ull;

    float rectSpacing =  2.f / mpi.size;
    float rectSize     = 1.f / mpi.size;
    float rectOffset  = -1.f + .25f*rectSize;
    
    float shiftPerDepth = .8f*rectSize / mpi.size;

    auto addBox = [&](int x, int y, int z, int dz)
    {
      float x0 = rectOffset + x * rectSpacing + z * shiftPerDepth;
      float y0 = rectOffset + y * rectSpacing + z * shiftPerDepth;
      float x1 = x0 + rectSize; 
      float y1 = y0 + rectSize;

      int i0 = vertices.size();
      vertices.push_back(vec3f(x0,y0,z+dz));
      vertices.push_back(vec3f(x0,y1,z+dz));
      vertices.push_back(vec3f(x1,y0,z+dz));
      vertices.push_back(vec3f(x1,y1,z+dz));
      
      indices.push_back(vec3i(i0)+vec3i(0,1,3));
      indices.push_back(vec3i(i0)+vec3i(0,3,2));
    };
    for (int z=0;z<mpi.size;z++)
      for (int y=0;y<mpi.size;y++)
        for (int x=0;x<mpi.size;x++) {
          size_t hash = 0x12345;
          hash = hash * FNV_PRIME ^ (x+123);
          hash = hash * FNV_PRIME ^ (y+456);
          int owner = (z + hash) % mpi.size;
          if (owner == mpi.rank) {
            addBox(x,y,z,0);
            addBox(x,y,z,mpi.size);
          }
        }
    // PING; PRINT(vertices.size()); 
    // for (auto v : vertices) PRINT(v);
    // PRINT(indices.size());
    // for (auto v : indices) PRINT(v);
    
    faceIteration::Context *fit
      = faceIteration::Context::init(gpuID,sizeof(UserMeshData),1,
                                     sizeof(PerLaunchData),
                                     devCode_ptx,
                                     "faceIterationCallback",
                                     "miniApp_perPixel");
    UserMeshData umd;
    umd.meshColor = owl::common::randomColor(mpi.rank+13);
    fit->setMesh(0,
                 vertices.data(),vertices.size(),
                 indices.data(),indices.size(),
                 &umd);
    // IMPORTANT: call built after all meshes have been set!
    fit->build();
    return fit;
  }
}
using namespace miniApp;


__global__ void localRender(vec2i fbSize,
                            Fragment *localFB,
                            Camera camera,
                            int rank,
                            int size)
{
  vec2i launchIdx;
  launchIdx.x = threadIdx.x+blockIdx.x*blockDim.x; if (launchIdx.x >= fbSize.x) return;
  launchIdx.y = threadIdx.y+blockIdx.y*blockDim.y; if (launchIdx.y >= fbSize.y) return;
  vec2i launchDims = fbSize;
  
  vec3f org = camera.org_00
    + (launchIdx.x+.5f)/launchDims.x*camera.org_du
    + (launchIdx.y+.5f)/launchDims.y*camera.org_dv;
  vec3f dir = camera.dir;
  // PerRayData prd;
  Fragment fragment;
  fragment.depth = INFINITY;
  fragment.opacity = 0.f;
  fragment.color = 0.f;

  int N = size;
  for (int iz=0;iz<N;iz++)
    for (int iy=0;iy<N;iy++)
      for (int ix=0;ix<N;ix++) {

        size_t FNV_PRIME = 0x00000100000001b3ull;
        
        float rectSpacing =  2.f / size;
        float rectSize     = 1.f / size;
        float rectOffset  = -1.f + .25f*rectSize;
        
        float shiftPerDepth = .8f*rectSize / size;
        
    //     auto addBox = [&](int x, int y, int z, int dz)
    // {
    //   float x0 = rectOffset + x * rectSpacing + z * shiftPerDepth;
    //   float y0 = rectOffset + y * rectSpacing + z * shiftPerDepth;
    //   float x1 = x0 + rectSize; 
    //   float y1 = y0 + rectSize;

    //   int i0 = vertices.size();
    //   vertices.push_back(vec3f(x0,y0,z+dz));
    //   vertices.push_back(vec3f(x0,y1,z+dz));
    //   vertices.push_back(vec3f(x1,y0,z+dz));
    //   vertices.push_back(vec3f(x1,y1,z+dz));
      
    //   indices.push_back(vec3i(i0)+vec3i(0,1,3));
    //   indices.push_back(vec3i(i0)+vec3i(0,3,2));
    // };

        auto boxTest = [&](float &t0, float &t1, box3f box)
        {
          vec3f t_lo = (box.lower - org) * rcp(dir);
          vec3f t_hi = (box.upper - org) * rcp(dir);
          vec3f t_nr = min(t_lo,t_hi);
          vec3f t_fr = max(t_lo,t_hi);
          t0 = max(t0,reduce_max(t_nr));
          t1 = min(t1,reduce_min(t_fr));
          return t0 <= t1;
        };
        auto anyHit = [&](float t) {
          vec3f color = owl::common::randomColor(rank+1);
          float opacity = .5f;
          
          fragment.color   += (1.f-fragment.opacity)*opacity*color;
          fragment.opacity += (1.f-fragment.opacity)*opacity;
        };
        auto intersectBox = [&](int x, int y, int z, int dz) {
          float x0 = rectOffset + x * rectSpacing + z * shiftPerDepth;
          float y0 = rectOffset + y * rectSpacing + z * shiftPerDepth;
          float x1 = x0 + rectSize; 
          float y1 = y0 + rectSize;
          float z0 = z+dz;
          float z1 = z0;
          box3f bb;
          bb.lower = { x0,y0,z0 };
          bb.upper = { x1,y1,z1 };

          float t0 = 0.f;
          float t1 = fragment.depth;
          if (boxTest(t0,t1,bb)) {
            anyHit(t0);
          }
        };
        size_t hash = 0x12345;
        hash = hash * FNV_PRIME ^ (ix+123);
        hash = hash * FNV_PRIME ^ (iy+456);
        int owner = (iz + hash) % size;
        if (owner == rank) {
          intersectBox(ix,iy,iz,0);
          intersectBox(ix,iy,iz,size);
        }
      }
  localFB[launchIdx.x+launchIdx.y*launchDims.x]
    = fragment;
}

/*! just for this miniapp - convert to an rgba image to save test image to png */
__global__
void resultToPixel(uint32_t *pixels, FinalCompositingResult *result, int numPixels)
{
  int tid = threadIdx.x+blockIdx.x*blockDim.x;
  if (tid >= numPixels) return;

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
  int rank,size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);
    
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
  // faceIteration::Context *fit = 
  //   setScene(comm,gpuID);
    
  // =============================================================================
  // set up a launch, and issue launch to render local frame buffer
  // =============================================================================
  Camera camera;
  // PerLaunchData launchData;
  // launchData.localFB = localFB;
  // launchData.rank = rank;
  
  // setCamera(launchData);
  // fit->launch(fbSize,&launchData);
  setCamera(camera);

  localRender<<<divRoundUp(fbSize,vec2i(8)),vec2i(8)>>>
    (fbSize,localFB,camera,rank,size);

  // =============================================================================
  // composite the local frame buffers
  // =============================================================================
  FinalCompositingResult *composited
    = comp->run();

  if (comp->mpi.rank == 0) {
    // rank 0 has the final coposited pixels, save the test image
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
  // delete fit;
  delete comp;
    
  MPI_Finalize();
  return 0;
}
  
