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
  };
  
  struct PerLaunchData : public faceIteration::LaunchData {
    vec3f camera_org;
    vec3f camera_d00;
    vec3f camera_dx;
    vec3f camera_dy;
    int frameID;
    Fragment *d_frameBuffer;
  };

  struct PerRayData {
    float depth;
    float accumulatedValue;
  }

}
