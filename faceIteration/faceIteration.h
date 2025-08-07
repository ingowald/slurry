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
    struct MeshData;
    
    // =============================================================================
    // APP-side interface
    // =============================================================================
  
    Context *init(int gpuID,
                  size_t sizeOfUserMesh,
                  int numMeshes,
                  size_t userLaunchDataSize,
                  const char *embeddedCode);
    void finalize(Context *ctx);
    void setMesh(int meshID,
                 const MeshData  *pUserMeshData);
    void render(vec2i launchDims, LaunchData *pLaunchData);
    



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
      // anything?
    };
    
    struct LaunchData {
      OptixTraversableHandle bvh;
    };

#ifdef __CUDA_ARCH__
# define FACE_ITERATION_DEFINE_PROGRAMS(perRayPerLaunchFct, perRayPerFaceFct) /* nothing yet */
#endif
  };
}
