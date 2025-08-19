// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "faceIteration.h"

namespace miniApp {
  using namespace slurry;

  struct UserMeshData : public faceIteration::MeshData {
    // anything you want to add ...
    vec3f meshColor;
  };

  struct Fragment {
    /*! each fragment type to be used with compositing:: library HAS
        to have a `float depth` member, but where, or what else is in
        that struct, is up to the app */
    float depth;

    /*! this is just for the sample, feel free to modify these as you
        wish: */
    vec3f color;
    float opacity;
  };

  struct FinalCompositingResult {
    vec3f color;
  };
  
  struct PerLaunchData : public faceIteration::LaunchData {
    struct {
      // for this simple example, let's use a orthogonal camera
      vec3f org_00;
      vec3f org_du;
      vec3f org_dv;
      vec3f dir;
    } camera;
    // only for debugging/illustration
    int rank;
    Fragment *localFB;
  };

  struct PerRayData : public faceIteration::PerRayData {
    Fragment fragment;
    bool dbg;
  };

}
