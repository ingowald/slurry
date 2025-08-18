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
                  const std::string &perFaceEntryPoint,
                  const std::string &perLaunchEntryPoint);
      ~ContextImpl() override;
      void build() override;
      void setMesh(int meshID,
                   const vec3f *vertices,
                   int numVertices,
                   const vec3i *indices,
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
                             const std::string &perFaceEntryPoint,
                             const std::string &perLaunchEntryPoint)
    {
      PING;
      owl = owlContextCreate(&gpuID,1);
      mod = owlModuleCreate(owl,embeddedPtxCode);
      
      std::string rgName = perLaunchEntryPoint;//+"_rg";
      rg = owlRayGenCreate(owl,mod,rgName.c_str(),0,0,0);

      OWLVarDecl lpArgs { "raw", OWLDataType(OWL_USER_TYPE_BEGIN+userLaunchDataSize), 0 };
      lp = owlParamsCreate(owl,(int)userLaunchDataSize,&lpArgs,1);

      OWLVarDecl gtArgs { "raw", OWLDataType(OWL_USER_TYPE_BEGIN+sizeOfUserMesh), 0 };
      gt = owlGeomTypeCreate(owl,OWL_GEOM_TRIANGLES,
                             (int)sizeOfUserMesh,&gtArgs,1);
      
      std::string chName = perFaceEntryPoint;//+"_ch";
      owlGeomTypeSetClosestHit(gt,0,mod,chName.c_str());
      
      std::string ahName = perFaceEntryPoint;//"_ah";
      owlGeomTypeSetAnyHit(gt,0,mod,ahName.c_str());
      
      PING;
      owlBuildPrograms(owl);
      owlBuildPipeline(owl);

      PING;
      meshes.resize(numMeshes);
    }

    ContextImpl::~ContextImpl()
    {
      owlContextDestroy(owl);
    }
    
    void ContextImpl::setMesh(int meshID,
                              const vec3f *vertices,
                              int numVertices,
                              const vec3i *indices,
                              int numIndices,
                              MeshData  *pUserMeshData)
    {
      assert(meshes[meshID] == 0);
      
      PING;
      OWLGeom geom = owlGeomCreate(owl,gt);
      
      PING;
      OWLBuffer verticesBuffer
        = owlDeviceBufferCreate(owl,OWL_FLOAT3,numVertices,vertices);
      owlTrianglesSetVertices(geom,verticesBuffer,numVertices,sizeof(vec3f),0);
      pUserMeshData->vertices = (vec3f*)owlBufferGetPointer(verticesBuffer,0);
      pUserMeshData->numVertices = numVertices;
      
      PING;
      OWLBuffer indicesBuffer
        = owlDeviceBufferCreate(owl,OWL_INT3,numIndices,indices);
      owlTrianglesSetIndices(geom,indicesBuffer,numIndices,sizeof(vec3i),0);
      pUserMeshData->indices = (vec3i*)owlBufferGetPointer(indicesBuffer,0);
      pUserMeshData->numIndices = numIndices;
      
      PING;
      owlGeomSetRaw(geom,"raw",pUserMeshData);
      PING;
      meshes[meshID] = geom;
    }
    
    void ContextImpl::build()
    {
      PING;
      for (auto mesh : meshes) assert(mesh);
      PING;
      blas = owlTrianglesGeomGroupCreate(owl,meshes.size(),meshes.data());
      owlGroupBuildAccel(blas);

      PING;
      tlas = owlInstanceGroupCreate(owl,1,&blas,nullptr);
      owlGroupBuildAccel(tlas);
      PING;
      owlBuildSBT(owl);
    }
    
    void ContextImpl::launch(vec2i launchDims, LaunchData *pLaunchData)
    {
      assert(tlas);
      PING;
      PRINT(tlas);
      PRINT(pLaunchData);
      pLaunchData->faceIt.bvh = owlGroupGetTraversable(tlas,0);
      PING;
      owlParamsSetRaw(lp,"raw",pLaunchData);
      PING;
      owlLaunch2D(rg,launchDims.x,launchDims.y,lp);
      PING;
    }
    

    Context *Context::init(int gpuID,
                           size_t sizeOfUserMesh,
                           int numMeshes,
                           size_t userLaunchDataSize,
                           const char *embeddedPtxCode,
                           const char *perFaceEntryPoint,
                           const char *perLaunchEntryPoint)
    {
      return new ContextImpl(gpuID,
                             sizeOfUserMesh,
                             numMeshes,
                             userLaunchDataSize,
                             embeddedPtxCode,
                             perFaceEntryPoint,
                             perLaunchEntryPoint);
    }
  }
}
