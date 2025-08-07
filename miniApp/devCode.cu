// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"

namespace slurry {
  
#if 1

  struct UserPRD {
    float accumulatedValue;
  };

  /*! this works just like a CUDA kernel, except you cannot specify any paramters */
  // OPTIX_RAYGEN_PROGRAM(slurryLaunch)()
  // {
  
  // }


  inline __device__ faceIteration::VisitResult rayHitsFace(UserPRD &prd)
  {
    return faceIteration::KEEP_TRAVERSING;
  }

  inline __device__ void launchOneRay(UserPRD &prd)
  {
  }

  SLURRY_DEFINE_PROGRAMS(launchOneRay,rayHitsFace);


#else
  struct UserPRD;
  inline __device__ void perFaceUserCode(UserPRD &prd);

  struct UserPRD;
  inline __device__ void perRayUserCode_initRay(UserPRD &prd);



  struct PerRayData : public faceIteration::PerRay {
    float    dist;
  };

  inline __device__ void rtFrontToBack(OptixTraversableHandle world,
                                       vec3f org,
                                       vec3f dir,
                                       float tMin,
                                       float tMax,
                                       slurry::PerRay &userPRD)
  {
    slurry::PerRay prd = { &userPRD, -1.f };
    float ray_tmax = ray.tmax;
    owl::Ray ray;
    ray.org = org;
    ray.dir = dir;
    ray.tmin = tMin;
    ray.tmax = tMax;
    while (1) {
    
      // ------------------------------------------------------------------
      // first, trace for ONLY closest hit - this will return the distance, or -1 in primID
      // ------------------------------------------------------------------
      prd.slurry.dist = -1.f; // mark as invalid
      ray.tmax = tMax;
      owl::traceRay(world,ray,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      if (prd.dist < 0.f)
        // no more surface(s) until ray_tmax
        break;
    
      ray.tmin = justBelow(prd.dist);
      ray.tmax = justAbove(prd.dist);
      owl::traceRay(world,ray,prd,OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
      if (prd.dist >= ray_tmax) /* ray is done */ break;
      
      ray.tmin = prd.dist;
    }
  }
#endif
}
