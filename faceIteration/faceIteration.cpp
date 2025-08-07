// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"

namespace slurry {
  namespace faceIteration {
    
    struct ContextImpl : public Context {
      ContextImpl(int gpuID,
                  size_t sizeOfUserMesh,
                  int numMeshes,
                  size_t userLaunchDataSize,
                  const char *embeddedPtxCode,
                  const std::string &entryPointName);
      ~ContextImpl() override;
      void finalize() override;
      void setMesh(int meshID,
                   const MeshData  *pUserMeshData) override;
      void launch(vec2i launchDims, LaunchData *pLaunchData) override;

      OWLModule mod = 0;
      OWLContext owl = 0;
      OWLRayGen rg = 0;
      OWLLaunchParams lp = 0;
      OWLGeomType gt = 0;
      OWLGroup tlas = 0;
    };

    ContextImpl::ContextImpl(int gpuID,
                             size_t sizeOfUserMesh,
                             int numMeshes,
                             size_t userLaunchDataSize,
                             const char *embeddedPtxCode,
                             const std::string &entryPointName)
    {
      owl = owlContextCreate(&gpuID,1);
      mod = owlModuleCreate(owl,embeddedPtxCode);
      
      std::string rgName = entryPointName+"_rg";
      rg = owlRayGenCreate(owl,mod,rgName.c_str(),0,0,0);

      OWLVarDecl lpArgs { "raw", OWL_USER_TYPE(userLaunchDataSize), 0 };
      lp = owlParamsCreate(owl,(int)userLaunchDataSize,&lpArgs,1);

      OWLVarDecl gtArgs { "raw", OWL_USER_TYPE(sizeOfUserMesh), 0 };
      gt = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                             (int)sizeOfUserMesh,&gtArgs,1);
      
      std::string chName = entryPointName+"_ch";
      owlGeomTypeSetClosestHit(gt,0,mod,chName.c_str());
      
      std::string ahName = entryPointName+"_ah";
      owlGeomTypeSetAnyHit(gt,0,mod,ahName.c_str());
      
      owlBuildPrograms(owl);
      owlBuildPipeline(owl);
    }

    ContextImpl::~ContextImpl()
    {
      owlContextDestroy(owl);
    }
    
    void ContextImpl::launch(vec2i launchDims, LaunchData *pLaunchData)
    {
      pLaunchData->faceIt.bvh = owlGroupGetTraversable(tlas,0);
      owlParamsSetRaw(lp,"raw",pLaunchData);
      owlLaunch2D(rg,launchDims.x,launchDims.y,lp);
    }
    

    Context *Context::init(int gpuID,
                           size_t sizeOfUserMesh,
                           int numMeshes,
                           size_t userLaunchDataSize,
                           const char *embeddedPtxCode,
                           const char *entryPointName)
    {
      return new ContextImpl(gpuID,
                             sizeOfUserMesh,
                             numMeshes,
                             userLaunchDataSize,
                             embeddedPtxCode,
                             entryPointName);
    }
  }
}
