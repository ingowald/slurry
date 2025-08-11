// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "compositing.h"
#include "miniApp.h"
#include <cuda_runtime.h>

/* has to match the name used in the embed_ptx cmake macro used in CMakeFile */
extern "C" const char devCode_ptx[];

namespace miniApp {

  typedef compositing::Context<Fragment,FinalCompositingResult> CompositingContext;
  vec2i fbSize { 800, 600 };
  struct {
    vec3f from { 0, 0, -1 };
    vec3f at   { 0, 0, 0 };
    vec3f up   { 0, 1, 0 };
    float fovy = 20.f;
  } camera;

  void setCamera(PerLaunchData &launchData)
  {
    /* vvvv all stolen from pete shirley's RTOW */
    const float vfov = camera.fovy;
    const vec3f vup = camera.up;
    const float aspect = fbSize.x / float(fbSize.y);
    const float theta = vfov * ((float)M_PI) / 180.0f;
    const float half_height = tanf(theta / 2.0f);
    const float half_width = aspect * half_height;
    const float focusDist = 10.f;
    const vec3f origin = camera.from;
    const vec3f w = normalize(camera.from - camera.at);
    const vec3f u = normalize(cross(vup, w));
    const vec3f v = cross(w, u);
    const vec3f lower_left_corner
      = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
    const vec3f horizontal = 2.0f*half_width *focusDist*u;
    const vec3f vertical   = 2.0f*half_height*focusDist*v;
    /* ^^^^ all stolen from pete shirley's RTOW */

    launchData.camera.org = origin;
    launchData.camera.dir_00 = lower_left_corner;
    launchData.camera.dir_dx = horizontal / fbSize.x;
    launchData.camera.dir_dy = vertical / fbSize.y;
  }

  __global__ void g_localCompositing(FinalCompositingResult *results,
                                     const Fragment *fragments_allRanksMyPixels,
                                     int numPixels,
                                     int numRanks)
  {
    int tid = threadIdx.x+blockIdx.x*blockDim.x;
    if (tid >= numPixels) return;
    
    const Fragment *myFragments = fragments_allRanksMyPixels + tid * numRanks;
    FinalCompositingResult *myResult = results+tid;
    myResult->value = 0.f;
    for (int depth=0;depth<numRanks;depth++)
      myResult->value += myFragments[depth].value;
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
  

  void setScene(faceIteration::Context *fit,
                const char *fileName);

  int main(int ac, char **av)
  {
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
    
    // =============================================================================
    // initialize out compositing context
    // =============================================================================
    CompositingContext *comp
      = new CompositingContext(MPI_COMM_WORLD,
                               localCompositing);
    Fragment *localFB = comp->resize(fbSize);

    // =============================================================================
    // specify the geometry
    // =============================================================================
    faceIteration::Context *fit
      = faceIteration::Context::init(gpuID,sizeof(UserMeshData),1,
                                     sizeof(PerLaunchData),
                                     devCode_ptx,
                                     "launchOneRay");
    setScene(fit,"test.binmesh");
    
    // =============================================================================
    // set up a launch, and issue launch to render local frame buffer
    // =============================================================================
    PerLaunchData launchData;
    launchData.localFB = localFB;
    setCamera(launchData);
    fit->launch(fbSize,&launchData);


    // =============================================================================
    // composite the local frame buffers
    // =============================================================================
    FinalCompositingResult *composited
      = comp->run();
    

    // =============================================================================
    // and wind down in reverse order
    // =============================================================================
    delete fit;
    delete comp;
    
    MPI_Finalize();
    return 0;
  }
  
}
