#pragma once

#include "faceIteration.h"

namespace miniApp {
  using namespace slurry;

  struct UserMeshData : public faceIteration::MeshData {
    float someDummyValue;
  };

  struct Fragment {
    float depth;
    float value;
    float alpha;
  };

  struct FinalCompositingResult {
    float value;
  };
  
  struct PerLaunchData : public faceIteration::LaunchData {
    struct {
      vec3f org;
      vec3f dir_00;
      vec3f dir_dx;
      vec3f dir_dy;
    } camera;
    int frameID;
    Fragment *d_frameBuffer;
  };

  struct PerRayData {
    float depth;
    float accumulatedValue;
  };

}
