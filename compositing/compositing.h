#pragma once

#include <owl/common/math/vec.h>
#include <mpi.h>

namespace slurry {
  namespace compositing {
    using owl::common::vec2i;
    
    /*! input is a pointer to an array of fragments (of the size the
      user indicated in compy::init) that this kernel is supposed to
      composite; sorted by pixel (first) and depth (second). The array
      contains `numPixelsThisRank*numRanks`, first all numRanks
      fragments for the first pixel (sorted by depth), then those for
      the second pixel, etc.

      Ie, assuming the user internally uses a "struct UserFragment", a
      given cuda thread `threadIdx` would find all its fragments as
      `((UserFragment*)inputFragments)[threadIdx*numPixelsOnThisRank+i],
      for i=0..numRanks, in front-to-back order
    */
    typedef void (*userCompositeKernel)(void *outputFragments,
                                        /*! one input fragment per pixel
                                          on this rank per rank in the
                                          scene, already sorted by
                                          depth */
                                        const void *inputFragments,
                                        int numPixelsThisRank,
                                        int numRanks);
  
    struct Context {
      virtual ~Context() = default;
      
      static Context *create(MPI_Comm comm,
                             size_t sizeOfUserInputFragmentType,
                             size_t sizeOfUserFinalCompositingResult);
    
      /*! resize context, return this rank's local write buffer */
      virtual void *resize(vec2i newSize) = 0;
    
      /*! run compositing, return final composited read buffer on rank 0,
        and nullptr on all other ranks */
      virtual void *run() = 0;
    };
  
  } // ::slurry::compositing
} // ::slurry
