// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"

namespace slurry {
  using namespace owl::common;
  // using owl::common::vec2i;
  // using owl::common::vec3i;
  // using owl::common::vec3f;
  // using owl::common::box3f;
  
  namespace faceIteration {
    
    struct Context;
    struct LaunchData;
    struct MeshData;
    
    // =============================================================================
    // APP-side interface
    // =============================================================================

    struct Context {
      virtual ~Context() = default;
      static Context *init(int gpuID,
                           size_t sizeOfUserMesh,
                           int numMeshes,
                           size_t userLaunchDataSize,
                           const char *embeddedPtxCode,
                           const char *perFaceEntryPoint,
                           const char *perLaunchEntryPoint);
      virtual void build() = 0;
      virtual void setMesh(int meshID,
                           const vec3f *vertices,
                           int numVertices,
                           const vec3i *indices,
                           int numIndices,
                           MeshData  *pUserMeshData) = 0;
      virtual void launch(vec2i launchDims, LaunchData *pLaunchData) = 0;
    };


    // =============================================================================
    // DEVICE programs-side interface
    // =============================================================================
    typedef enum { DONE_TRAVERSING=0, KEEP_TRAVERSING=1 } VisitResult;

    /*! BASE class for all per-mesh data; it's the user's job to add
        more stuff if required by subclassing */
    struct MeshData {
      vec3f *vertices;
      vec3i *indices;
      int numVertices;
      int numIndices;
    };
    
    struct PerRayData {
      struct {
        float tHit;
      } faceIt;
    };
    
    struct LaunchData {
      struct {
        OptixTraversableHandle bvh;
      } faceIt;
    };

#ifdef __CUDACC__
    inline __device__
    void traceFrontToBack(OptixTraversableHandle bvh,
                          vec3f org, vec3f dir, float t0, float t1,
                          PerRayData &prd);
    
// # define FACE_ITERATION_DEFINE_PROGRAMS(perRayPerLaunchFct, perRayPerFaceFct) /* nothing yet */
#endif



    // #############################################################################
    // IMPLEMENTATION
    // #############################################################################

#ifdef __CUDA_ARCH__
    inline __device__ float justBelow(float f)
    { return nextafterf(f,-1.f); }
    inline __device__ float justAbove(float f)
    { return nextafterf(f,INFINITY); }
    
    inline __device__
    void traceFrontToBack(OptixTraversableHandle tlas,
                          vec3f org, vec3f dir, float t0, float t1,
                          PerRayData &prd)
    {
      // slurry::PerRay prd = { &userPRD, -1.f };
      owl::Ray ray;
      ray.origin = org;
      ray.direction = dir;
      ray.tmin = t0;
      ray.tmax = t1;
      while (1) {
        // ------------------------------------------------------------------
        // first, trace for ONLY closest hit - this will return the
        // distance, or -1 in faceIt.tHit
        // ------------------------------------------------------------------
        prd.faceIt.tHit = -1.f; // mark as invalid
        ray.tmax = t1;
        owl::traceRay(tlas,ray,prd,OPTIX_RAY_FLAG_DISABLE_ANYHIT);
        if (prd.faceIt.tHit < 0.f)
          // no more surface(s) until ray_tmax
          break;
    
        ray.tmin = justBelow(prd.faceIt.tHit);
        ray.tmax = justAbove(prd.faceIt.tHit);
        owl::traceRay(tlas,ray,prd,OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
      
        ray.tmin = prd.faceIt.tHit;
      }
  }
    
#endif
  }
}
