#pragma once

#include "faceIteration.h"

namespace miniApp {
  using namespace slurry;

  struct UserMeshData : public faceIteration::MeshData {
    // anything you want to add ...
    vec3f meshColor;
  };

  struct Fragment {
    float depth;
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
    int frameID;
    Fragment *localFB;
  };

  struct PerRayData : public faceIteration::PerRayData {
    Fragment fragment;
    bool dbg;
  };

}
