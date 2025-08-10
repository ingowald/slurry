// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#include "faceIteration.h"
#include <vector>

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
      void build() override;
      void setMesh(int meshID,
                   const vec3f *vertices,
                   int numVertices,
                   const vec3f *indices,
                   int numIndices,
                   MeshData  *pUserMeshData) override;
      void launch(vec2i launchDims, LaunchData *pLaunchData) override;

      std::vector<OWLGeom> meshes;
      OWLModule mod = 0;
      OWLContext owl = 0;
      OWLRayGen rg = 0;
      OWLLaunchParams lp = 0;
      OWLGeomType gt = 0;
      OWLGroup blas = 0;
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

      meshes.resize(numMeshes);
    }

    ContextImpl::~ContextImpl()
    {
      owlContextDestroy(owl);
    }
    
    void ContextImpl::setMesh(int meshID,
                              const vec3f *vertices,
                              int numVertices,
                              const vec3f *indices,
                              int numIndices,
                              MeshData  *pUserMeshData)
    {
      OWLGeom geom = owlGeomCreate(owl,gt);
      
      OWLBuffer verticesBuffer
        = owlDeviceBufferCreate(owl,OWL_FLOAT3,numVertices,vertices);
      owlTrianglesSetVertices(geom,verticesBuffer,numVertices,sizeof(vec3f),0);
      pUserMeshData->vertices = (vec3f*)owlBufferGetPointer(verticesBuffer,0);
      pUserMeshData->numVertices = numVertices;
      
      OWLBuffer indicesBuffer
        = owlDeviceBufferCreate(owl,OWL_INT3,numIndices,indices);
      owlTrianglesSetIndices(geom,indicesBuffer,numIndices,sizeof(vec3i),0);
      pUserMeshData->indices = (vec3i*)owlBufferGetPointer(indicesBuffer,0);
      pUserMeshData->numIndices = numIndices;
      
      owlGeomSetRaw(geom,"raw",pUserMeshData);
      assert(meshes[meshID] == 0);
      meshes[meshID] = geom;
    }
    
    void ContextImpl::build()
    {
      for (auto mesh : meshes) assert(mesh);
      blas = owlTrianglesGeomGroupCreate(owl,meshes.size(),meshes.data());
      owlGroupBuildAccel(blas);

      tlas = owlInstanceGroupCreate(owl,1,&blas,nullptr);
      owlGroupBuildAccel(tlas);
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
