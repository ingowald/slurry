#include "compositing.h"

namespace slurry {
  namespace compositing {

    struct ContextImpl : public Context {
      ContextImpl(MPI_Comm comm,
                  size_t sizeOfUserInputFragmentType,
                  size_t sizeOfUserFinalCompositingResult);
      ~ContextImpl() override;
      
      /*! resize context, return this rank's local write buffer */
      void *resize(vec2i newSize) override;
    
      /*! run compositing, return final composited read buffer on rank 0,
        and nullptr on all other ranks */
      void *run() override;
      
      const MPI_Comm comm;
      int rank = -1;
      int size = -1;
      const size_t sizeOfUserInputFragmentType;
      const size_t sizeOfUserFinalCompositingResult;
      vec2i fbSize{-1,-1};
      int begin = -1;
      int end = -1;
      int numMyPixels = -1;
      int numTotalFragments = -1;
      void *myFragments = 0;
      void *allFragments = 0;
      void *compositedResult = 0;
    };
    
    Context *Context::create(MPI_Comm comm,
                             size_t sizeOfUserInputFragmentType,
                             size_t sizeOfUserFinalCompositingResult)
    {
      
      return new ContextImpl(comm,
                             sizeOfUserInputFragmentType,
                             sizeOfUserFinalCompositingResult);
    }

    ContextImpl::ContextImpl(MPI_Comm comm,
                             size_t sizeOfUserInputFragmentType,
                             size_t sizeOfUserFinalCompositingResult)
      : comm(comm),
        sizeOfUserInputFragmentType(sizeOfUserInputFragmentType),
        sizeOfUserFinalCompositingResult(sizeOfUserFinalCompositingResult)
    {
      MPI_Comm_rank(comm,&rank);
      MPI_Comm_size(comm,&size);
    }

    ContextImpl::~ContextImpl()
    {
      if (allFragments) cudaFree(allFragments);
      if (myFragments) cudaFree(myFragments);
      if (compositedResult) cudaFree(compositedResult);
    }

    /*! resize context, return this rank's local write buffer */
    void *ContextImpl::resize(vec2i newSize)
    {
      this->fbSize = newSize;
      int numPixelsTotal = fbSize.x*fbSize.y;
      this->begin = numPixelsTotal * (rank+0) / size;
      this->end   = numPixelsTotal * (rank+1) / size;
      this->numMyPixels = end - begin;
      this->numTotalFragments = numMyPixels * size;
      
      if (allFragments) cudaFree(allFragments);
      cudaMalloc((void **)&allFragments,numTotalFragments*sizeOfUserInputFragmentType);
      if (myFragments) cudaFree(myFragments);
      cudaMalloc((void **)&myFragments,fbSize.x*fbSize.y*sizeOfUserInputFragmentType);

      if (compositedResult) cudaFree(compositedResult);
      if (rank == 0)
        cudaMalloc(&compositedResult,fbSize.x*fbSize.y*sizeOfUserFinalCompositingResult);
      
      return myFragments;
    }
    
    
  } // ::slurry::compositing
} // ::slurry

