# slurry - A Sample Code for how to Combine (OptiX) Ray Tracing with MPI Compositing

This codebase provides a simple example - and some possibly
useful-byond-this-sample helper libraries - for OptiX-based (per-node)
ray tracing with MPI Compositing.

In particular, it contains three distinct components:

## I) `faceIteration` - a simple OptiX-based tool for iterating through
  multiple successive triangular faces along a given ray. This tool
  allows to specify 
  
  a) (on the host) a model of one or more triangle meshes
  
  b) a (device-side) "callback" function to be called on a
  (user-specified) struct for every triangle facet along a given ray.
  What this function does---what data it carries per ray, and what it
  does to this data when it hits a triangle---is totally up to the user.
  
  c) a way to launch such rays in completely user-defined way, simialr
  to a CUDA kernel launch.
  
## II) `compositing` - a helper library for MPI-based (CUDA-side)
compositing. This library is templated both over what a "fragment" is
(ie, which kind of data each rank produces for each pixel), and what
the "final compositing result" for each (final) pixel is (ie, what
those fragments eventually end up as _after_ they have been
composited), and also allows the user to specify his own callback for
compositing a set of differnet ranks' depth-sorted fragments that this
library has gathered/rearranged/sorted/etc). 

As an example, a user could define a "fragment" to be
'color+depth+opacity', the final compositing result to be a 'color',
and provide a cuda kernel that implements (for example, but not
limited to!) standard 'over' compositing. The library itself then
provides the user with an array where to write his fragments to, and a
`Compositing::run()` function that fully automatically exchanges
fragments with other ranks, rearranges and depth-sorts them, calls (on
each rank) the user's compositing kernel to composite some of these,
merges and re-arranges the results, and returns to the user a array of
final composited results on rank 0.

## III) A mini-app that demonstrates this for simple alpha-transparency

This mini-app defines a Fragment as color+alpha+depth, and compositing
as over-compositing. It uses `faceIteration` to traverse through a
given rank's (semi-transparent) triangles to compute a fragment for
each pixel (on each rank), then uses the compositing library to do
MPI-parallel compositing of those ranks' fragments into a final image.


To customize this to your own needs: you should not need to modify
anything in either `compositing/` or `faceIteration/`, but feel free
to base your code off the miniApp, and ...

- e.g., change the `Fragment` and/or `FinalCompositingResult` type(s)
  in `miniApp/miniApp.h` to whatever you want to composite from and/or into.
  Be aware that your Fragment type has to be derived from `slurry::
  
- change the `g_localCompositing()` (CUDA-)kernel in
  `miniApp/miniApp.cu` to change how fragments are composited. For N
  ranks, you can expect this kernel to be called with N depth-sorted
  fragments per launch index, do with those as you wish, as long as
  you produce whatever `FinalCompositingResult` type you have defined.
  
- change the `UserMeshData` (`miniApp/miniApp.h`) to any other data
  you might want to pass along to the device for each triangle mesh
  (eg, to have different materials/opacities for each mesh, or
  additional per-triangle/per-vertex data arrays, etcpp).

- change the `inline __device__ faceIteration::VisitResult
  miniApp::faceIterationCallback()` to change what fragment(s) you
  want to produce on each rank. You can freely change what "per ray
  data" you want to carry along your ray, what to do in this callback,
  and whether you want this callback to stop after this triangle
  (`return faceIteration::TERMINATE_TRAVERSAL`) or continue
  traversring to the next facet (`return
  faceIteration::KEEP_TRAVERSING`). Facets are guaranteed to be
  visited in sorted depth order.

