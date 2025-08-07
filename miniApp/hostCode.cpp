
#include "compositing.h"
#include "miniApp.h"
#include <cuda_runtime.h>

/* has to match the name used in the embed_ptx cmake macro used in CMakeFile */
extern "C" const char embedded_devCode_ptx[];

namespace miniApp {

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


  void setScene(faceIteration::Context *fit,
                const char *fileName);
  
  extern "C" int main(int ac, char **av)
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
    compositing::Context *comp
      = compositing::Context::create(MPI_COMM_WORLD,
                                     sizeof(Fragment),
                                     sizeof(FinalCompositingResult));
    comp->resize(fbSize.x,fbSize.y);

    // =============================================================================
    // specify the geometry
    // =============================================================================
    faceIteration::Context *fit
      = faceIteration::Context::init(gpuID,sizeof(UserMeshData),1,
                                     sizeof(PerLaunchData),
                                     embedded_devCode_ptx,
                                     "launchOneRay");
    setScene(fit,"test.binmesh");
    
    // =============================================================================
    // set up a launch
    // =============================================================================
    PerLaunchData launchData;
    setCamera(launchData);



    

    // =============================================================================
    // and wind down in reverse order
    // =============================================================================
    delete fit;
    delete comp;
    
    MPI_Finalize();
    return 0;
  }
  
}
