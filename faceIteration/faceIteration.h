// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "owl/owl.h"
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"

namespace slurry {
  using owl::common::vec2i;
  using owl::common::vec3i;
  using owl::common::vec3f;
  using owl::common::box3f;
  
  namespace faceIteration {
    
    struct Context;
    struct LaunchData;

    // =============================================================================
    // APP-side interface
    // =============================================================================
  
    Context *init(int gpuID,
                  int numMeshes,
                  size_t userLaunchDataSize,
                  const char *embeddedCode);
    void finalize(Context *ctx);
    void setMesh(int meshID,
                 const vec3f *vertices, int numVertices,
                 const vec3i *indices, int numIndices,
                 const void  *userDataPointerOnGPU = 0);
    void render(vec2i launchDims, LaunchData *pLaunchData);




    // =============================================================================
    // DEVICE programs-side interface
    // =============================================================================
    typedef enum { DONE_TRAVERSING=0, KEEP_TRAVERSING=1 } VisitResult;
  
    struct PerRay {
      // anything?
    };
    struct LaunchData {
      OptixTraversableHandle bvh;
    };

#ifdef __CUDA_ARCH__
# define SLURRY_DEFINE_PROGRAMS(perRayPerLaunchFct, perRayPerFaceFct) /* nothing yet */
#endif
  };
}
