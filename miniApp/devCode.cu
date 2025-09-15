// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"
#include "miniApp.h"

DECLARE_OPTIX_LAUNCH_PARAMS(miniApp::PerLaunchData);


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

    vec3f org = optixGetWorldRayOrigin();
    if (prd.dbg) printf("face it hit at t=%f z=%f\n",
                        optixGetRayTmax(),
                        org.z+optixGetRayTmax());
    
    vec3f dir = optixGetWorldRayDirection();
    dir = normalize(dir);
    prd.fragment.depth = min(prd.fragment.depth,optixGetRayTmax());
    vec3f color = mesh.meshColor;
    float opacity = .5f;

    prd.fragment.color   += (1.f-prd.fragment.opacity)*opacity*color;
    prd.fragment.opacity += (1.f-prd.fragment.opacity)*opacity;

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
    PerRayData prd;
    prd.dbg = (launchIdx == launchDims/2);
    if (prd.dbg)
      printf("DBG AT (%i %i) out of (%i %i)\n",
             launchIdx.x,
             launchIdx.y,
             launchDims.x,
             launchDims.y);
    prd.fragment.depth = INFINITY;
    prd.fragment.opacity = 0.f;
    prd.fragment.color = 0.f;
    if (prd.dbg)
      printf("launching ray (%f %f %f)+t*(%f %f %f)\n",
             org.x,
             org.y,
             org.z,
             dir.x,
             dir.y,
             dir.z);

    faceIteration::traceFrontToBack(launchData.faceIt.bvh,
                                    org,dir,0.f,INFINITY,
                                    prd,prd.dbg);
   launchData.localFB[launchIdx.x+launchIdx.y*launchDims.x]
      = prd.fragment;
  }

}
