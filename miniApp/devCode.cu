// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"
#include "PerLaunch.h"

DECLARE_OPTIX_LAUNCH_PARAMS(miniApp::PerLaunchData);

namespace miniApp {

  inline __device__ faceIteration::VisitResult rayHitsFace()
  {
    const UserMesh &mesh = owl::getProgramData<UserMesh>();
    
    int primID = optixGetPrimitiveIndex();
    vec3i idx = mesh.indices[primID];
    vec3f a = mesh.vertices[idx.x];
    vec3f b = mesh.vertices[idx.y];
    vec3f c = mesh.vertices[idx.z];
    vec3f N = normalize(cross(b-a,c-a));
    
    vec3f dir = normalize(optixGetWorldRayDirection());
    prd.accumulatedValue += mesh.someDummyValue*fabsf(dot(N,dir));
    
    return faceIteration::KEEP_TRAVERSING;
  }

  /*! this works just like a cuda kernel, you just can't directly pass
      any paramters; they have to go through the "LaunchParams"
      abstraction */
  inline __device__ void launchOneRay()
  {
    vec2i launchIdx = optixGetLaunchIndex();
    vec2i launchDims = optixGetLaunchDimensions();
    if (launchIdx.x >= launchDims.x || launchIdx.y >= launchDims.y) return;
    
    const PerLaunchData &launchData
      = /* this is a device global that gets filled in by optix, just use it */
      optixLaunchParams;

    vec3f org = launchData.camera_org;
    vec3f dir = normalize(launchData.camera_d00;
                          + (launchIdx.x+.5f)*launchData.camera_du
                          + (launchIdx.y+.5f)*launchData.camera_dv);
    owl::Ray ray(org,dir);

    PerRayData perRayData;
    owl::traceRay(ray,perRayData);
  }

  FACE_ITERATION_DEFINE_PROGRAMS(launchOneRay,rayHitsFace);


#if 0
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
