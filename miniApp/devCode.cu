// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"
#include "miniApp.h"

DECLARE_OPTIX_LAUNCH_PARAMS(miniApp::PerLaunchData);

namespace miniApp {
  
  inline __device__ faceIteration::VisitResult rayHitsFace()
  {
    const UserMeshData &mesh = owl::getProgramData<UserMeshData>();
    PerRayData &prd = owl::getPRD<PerRayData>();
    
    int primID = optixGetPrimitiveIndex();
    vec3i idx = mesh.indices[primID];
    vec3f a = mesh.vertices[idx.x];
    vec3f b = mesh.vertices[idx.y];
    vec3f c = mesh.vertices[idx.z];
    vec3f N = normalize(cross(b-a,c-a));
    
    vec3f dir = optixGetWorldRayDirection();
    dir = normalize(dir);
    prd.fragment.depth = min(prd.fragment.depth,optixGetRayTmax());
    float oldOpacity = prd.fragment.opacity;
    float thisOpacity = .1f;
    
    prd.fragment.value += (1.f-oldOpacity)*thisOpacity*mesh.someDummyValue*fabsf(dot(N,dir));
    prd.fragment.opacity += (1.f-oldOpacity)*thisOpacity;
      
    return faceIteration::KEEP_TRAVERSING;
  }

  /*! this works just like a cuda kernel, you just can't directly pass
      any paramters; they have to go through the "LaunchParams"
      abstraction */
  OPTIX_RAYGEN_PROGRAM(launchOneRay)()
  {
    vec3i launchIdx = optixGetLaunchIndex();
    vec3i launchDims = optixGetLaunchDimensions();
    if (launchIdx.x >= launchDims.x || launchIdx.y >= launchDims.y) return;
    
    const PerLaunchData &launchData
      = /* this is a device global that gets filled in by optix, just use it */
      optixLaunchParams;

    vec3f org = launchData.camera.org;
    vec3f dir = normalize(launchData.camera.dir_00
                          + (launchIdx.x+.5f)*launchData.camera.dir_dx
                          + (launchIdx.y+.5f)*launchData.camera.dir_dy);
    PerRayData prd;
    prd.fragment.depth = INFINITY;
    prd.fragment.opacity = 0.f;
    prd.fragment.value = 0.f;
    
    faceIteration::traceFrontToBack(launchData.faceIt.bvh,
                                    org,dir,0.f,INFINITY,
                                    prd);

    launchData.d_perRankFragments[launchIdx.x+launchDims.x*launchIdx.y]
      = prd.fragment;
  }



#if 0
  FACE_ITERATION_DEFINE_PROGRAMS(launchOneRay,rayHitsFace);
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
