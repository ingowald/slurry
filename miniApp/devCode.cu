// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"
#include "miniApp.h"

DECLARE_OPTIX_LAUNCH_PARAMS(miniApp::PerLaunchData);

#define FACEIT_PER_FACE_FUNCTION(userFct)                         \
  OPTIX_CLOSEST_HIT_PROGRAM(userFct)()                            \
  {                                                               \
    ::slurry::faceIteration::PerRayData &prd                      \
      = owl::getPRD<slurry::faceIteration::PerRayData>();         \
    auto result = userFct();                                      \
    if (result == ::slurry::faceIteration::DONE_TRAVERSING)       \
      prd.faceIt.tHit = -1.f;                                            \
  }                                                               \
  OPTIX_ANY_HIT_PROGRAM(userFct)()                                \
  {                                                               \
    ::slurry::faceIteration::PerRayData &prd                      \
      = owl::getPRD<slurry::faceIteration::PerRayData>();         \
    auto result = userFct();                                      \
    if (result == ::slurry::faceIteration::DONE_TRAVERSING)       \
      prd.faceIt.tHit = -1.f;                                            \
  }


namespace miniApp {
  inline __device__ faceIteration::VisitResult faceIterationCallback()
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
    vec3f color = mesh.meshColor;
    float opacity = .5f;

    prd.fragment.color   += (1.f-prd.fragment.opacity)*opacity*color;
    prd.fragment.opacity += (1.f-prd.fragment.opacity)*opacity;

    if (prd.dbg) {
      printf("depth %f\n",prd.fragment.depth);
      printf("color %f %f %f\n",color.x,color.y,color.z);
    }
    return faceIteration::KEEP_TRAVERSING;
  }
  FACEIT_PER_FACE_FUNCTION(faceIterationCallback);

  /*! this works just like a cuda kernel, you just can't directly pass
      any paramters; they have to go through the "LaunchParams"
      abstraction */
  OPTIX_RAYGEN_PROGRAM(miniApp_perPixel)()
  {
    vec3i launchIdx = optixGetLaunchIndex();
    vec3i launchDims = optixGetLaunchDimensions();
    if (launchIdx.x >= launchDims.x || launchIdx.y >= launchDims.y) return;

    const PerLaunchData &launchData
      = /* this is a device global that gets filled in by optix, just use it */
      optixLaunchParams;

    vec3f org = launchData.camera.org_00
      + (launchIdx.x+.5f)/launchDims.x*launchData.camera.org_du
      + (launchIdx.y+.5f)/launchDims.y*launchData.camera.org_dv;
    vec3f dir = launchData.camera.dir;
    // vec3f org = launchData.camera.org;
    // vec3f dir = normalize(launchData.camera.dir_00
    //                       + (launchIdx.x+.5f)*launchData.camera.dir_dx
    //                       + (launchIdx.y+.5f)*launchData.camera.dir_dy);
    PerRayData prd;
    prd.dbg = (launchIdx == launchDims/2);
    prd.fragment.depth = INFINITY;
    prd.fragment.opacity = 0.f;
    prd.fragment.color = 0.f;

    if (prd.dbg) printf("tracing...\n");
    faceIteration::traceFrontToBack(launchData.faceIt.bvh,
                                    org,dir,0.f,INFINITY,
                                    prd);

    launchData.localFB[launchIdx.x+launchDims.x*launchIdx.y]
      = prd.fragment;
  }

}
