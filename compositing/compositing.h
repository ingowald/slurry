// Copyright 2025 Ingo Wald
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <owl/common/math/vec.h>
#include <mpi.h>
#include <vector>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

namespace slurry {
  namespace compositing {
    using namespace owl::common;

    template<typename FragmentT, typename ResultT>
    struct Context {
      Context(MPI_Comm comm,
              /*! user-provided kernel that can composite a range of
                pixels' worth of fragments. input is a array of
                fragments with 'numPixelsThisRank' pixels, and
                'numRanks' fragments per pixel; order is first all
                numRanks fragments for the first pixel (sorted by
                depth), then those for the second pixel, etc.
              */
              void (*userCompositingKernel)(ResultT *outputFragments,
                                          /*! one input fragment per pixel on
                                            this rank per rank in the scene,
                                            already sorted by depth */
                                          const FragmentT *inputFragments,
                                          int numPixelsThisRank,
                                          int numRanks)
              );
      ~Context();
      
      /*! resize context, return this rank's local write buffer */
      FragmentT *resize(vec2i newSize);
      
      /*! run compositing, return final composited read buffer on rank 0,
        and nullptr on all other ranks */
      ResultT *run();
      
      struct {
        MPI_Comm comm;
        int rank = -1;
        int size = -1;
      } mpi;
              /*! user-provided kernel that can composite a range of
                pixels' worth of fragments. input is a array of
                fragments with 'numPixelsThisRank' pixels, and
                'numRanks' fragments per pixel; order is first all
                numRanks fragments for the first pixel (sorted by
                depth), then those for the second pixel, etc.
              */
      void (*userCompositingKernel)(ResultT *outputFragments,
                                  const FragmentT *inputFragments,
                                  int numPixelsThisRank,
                                  int numRanks) = nullptr;
      vec2i fbSize{-1,-1};
      int my_begin = -1;
      int my_end = -1;
      int numPixelsInFrame   = -1;
      FragmentT *myFragments  = 0;
      FragmentT *allFragments = 0;
      uint64_t  *sortKeys     = 0;
      ResultT   *localResults = 0;
      ResultT   *finalResults = 0;
    };

    // =============================================================================
    // IMPLEMENTATION
    // =============================================================================
    
    template<typename FragmentT, typename ResultT>
    Context<FragmentT,ResultT>::Context
    (MPI_Comm comm,
     void (*userCompositingKernel)(ResultT *outputFragments,
                                 /*! one input fragment per pixel on
                                   this rank per rank in the scene,
                                   already sorted by depth */
                                 const FragmentT *inputFragments,
                                 int numPixelsThisRank,
                                 int numRanks)
     )
      : userCompositingKernel(userCompositingKernel)
    {
      mpi.comm = comm;
      MPI_Comm_rank(mpi.comm,&mpi.rank);
      MPI_Comm_size(mpi.comm,&mpi.size);
    }

    template<typename FragmentT, typename ResultT>
    Context<FragmentT,ResultT>::~Context()
    {
      if (allFragments) cudaFree(allFragments);
      if (myFragments) cudaFree(myFragments);
      if (sortKeys) cudaFree(sortKeys);
      if (localResults) cudaFree(localResults);
      if (finalResults) cudaFree(finalResults);
    }

    /*! resize context, return this rank's local write buffer */
    template<typename FragmentT, typename ResultT>
    FragmentT *Context<FragmentT,ResultT>::resize(vec2i newSize)
    {
      this->fbSize = newSize;
      numPixelsInFrame = fbSize.x*fbSize.y;
      this->my_begin = numPixelsInFrame * (mpi.rank+0) / mpi.size;
      this->my_end   = numPixelsInFrame * (mpi.rank+1) / mpi.size;
      
      if (allFragments) cudaFree(allFragments);
      if (localResults) cudaFree(localResults);
      if (finalResults) cudaFree(finalResults);
      if (myFragments)  cudaFree(myFragments);
      if (sortKeys)     cudaFree(sortKeys);

      // ------------------------------------------------------------------
      // INPUT fragments are one per pixel
      cudaMalloc((void **)&myFragments, (fbSize.x*fbSize.y)*sizeof(FragmentT));

      // ------------------------------------------------------------------
      // COMPOSITING inputs (after DPS) is my my_begin...my_end range...

      // ... sort keys is one per mine per rank
      cudaMalloc((void **)&sortKeys,    (my_end-my_begin)*mpi.size*sizeof(uint64_t));
      // ... received fragments si one per mine per rank
      cudaMalloc((void **)&allFragments,(my_end-my_begin)*mpi.size*sizeof(FragmentT));
      // ... local results is one per mine, period
      cudaMalloc((void **)&localResults,(my_end-my_begin)*sizeof(ResultT));
      
      
      // ------------------------------------------------------------------
      // FINAL result is one per pixel
      if (mpi.rank == 0)
        cudaMalloc((void **)&finalResults,fbSize.x*fbSize.y*sizeof(ResultT));
      
      return myFragments;
    }
    
    template<typename FragmentT>
    void sortFragments(uint64_t *keys, FragmentT *fragments, size_t numFragments)
    {
      // Determine temporary device storage requirements
      void     *d_temp_storage = nullptr;
      size_t   temp_storage_bytes = 0;
      uint64_t *d_keys_in = keys;
      uint64_t *d_keys_out = 0;
      cudaMalloc((void**)&d_keys_out,numFragments*sizeof(uint64_t));
      FragmentT *d_values_in = fragments;
      FragmentT *d_values_out = 0;
      cudaMalloc((void**)&d_values_out,numFragments*sizeof(FragmentT));
      
      int num_items = numFragments;
      cub::DeviceRadixSort::SortPairs
        (d_temp_storage, temp_storage_bytes,
         d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);

      // Allocate temporary storage
      cudaMalloc(&d_temp_storage, temp_storage_bytes);

      // Run sorting operation
      cub::DeviceRadixSort::SortPairs
        (d_temp_storage, temp_storage_bytes,
         d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
      cudaMemcpy(d_values_in,d_values_out,numFragments*sizeof(FragmentT),
                 cudaMemcpyDefault);
      cudaFree(d_keys_out);
      cudaFree(d_values_out);
      cudaFree(d_temp_storage);
    }
    
      
    template<typename FragmentT>
    __global__
    void g_generateKeys(uint64_t *sortKeys,
                        FragmentT *fragments,
                        int numFragsThisRank,
                        int numRanks)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numRanks*numFragsThisRank) return;

      int pixelID = tid % numFragsThisRank;
      // int rank    = tid / numFragsThisRank;

      uint64_t key
        = (uint64_t(pixelID) << 32)
        | uint64_t(__float_as_int(fragments[tid].depth))
        ;
      sortKeys[tid] = key;
    }
    
    template<typename FragmentT, typename ResultT>
    ResultT *Context<FragmentT,ResultT>::run()
    {
      // =============================================================================
      // gather all the fragments for my_begin..my_range, from every rank
      // =============================================================================
      void *sendBuf = myFragments;
      void *recvBuf = allFragments;
      std::vector<int> sendOffsets(mpi.size);
      std::vector<int> sendCounts(mpi.size);
      std::vector<int> recvOffsets(mpi.size);
      std::vector<int> recvCounts(mpi.size);
      for (int r=0;r<mpi.size;r++) {
        int r_begin = (r+0) * numPixelsInFrame / mpi.size;
        int r_end   = (r+1) * numPixelsInFrame / mpi.size;
        sendOffsets[r] = r_begin * sizeof(FragmentT);
        sendCounts[r]  = (r_end-r_begin) * sizeof(FragmentT);

        recvOffsets[r] = (my_end-my_begin) * r * sizeof(FragmentT);
        recvCounts[r] = (my_end-my_begin) * sizeof(FragmentT);
      }
      MPI_Alltoallv(sendBuf,
                    (const int*)sendCounts.data(),
                    (const int*)sendOffsets.data(),
                    MPI_BYTE,
                    recvBuf,
                    (const int*)recvCounts.data(),
                    (const int*)recvOffsets.data(),
                    MPI_BYTE,
                    mpi.comm);

      
      // =============================================================================
      // sort these fragments by pixel ID and depth
      // =============================================================================
      int numWorkItems = (my_end-my_begin)*mpi.size;
      int bs = 1024;
      int nb = divRoundUp(numWorkItems,bs);
      uint64_t *sortKeys;
      cudaMalloc((void**)&sortKeys,numWorkItems*sizeof(uint64_t));
      g_generateKeys<<<nb,bs>>>(sortKeys,allFragments,my_end-my_begin,mpi.size);
      sortFragments(sortKeys,allFragments,numWorkItems);
      cudaFree(sortKeys);
      // =============================================================================
      // let user composite these fragments, and turn then into results
      // =============================================================================
      this->userCompositingKernel(localResults,allFragments,my_end-my_begin,mpi.size);
      cudaStreamSynchronize(0);

      // =============================================================================
      // and gather them all at rank 0
      // =============================================================================
      sendBuf = localResults;
      recvBuf
        = (mpi.rank == 0)
        ? /* the actual results */finalResults
        : /* anything that's not null */localResults;
      for (int r=0;r<mpi.size;r++) {
        if (mpi.rank == 0) {
          int r_begin = (r+0) * numPixelsInFrame / mpi.size;
          int r_end   = (r+1) * numPixelsInFrame / mpi.size;
          recvOffsets[r] = r_begin * sizeof(ResultT);
          recvCounts[r]  = (r_end-r_begin) * sizeof(ResultT);
        } else {
          recvOffsets[r] = 0;
          recvCounts[r] = 0;
        };
        sendOffsets[r] = 0;
        sendCounts[r]
          = (r == 0)
          ? (my_end-my_begin)*sizeof(ResultT)
          : 0;
      }
      MPI_Alltoallv(sendBuf,
                (const int*)sendCounts.data(),
                (const int*)sendOffsets.data(),
                MPI_BYTE,
                recvBuf,
                (const int*)recvCounts.data(),
                (const int*)recvOffsets.data(),
                MPI_BYTE,
                mpi.comm);
      return finalResults;
    }
    
  } // ::slurry::compositing
} // ::slurry
