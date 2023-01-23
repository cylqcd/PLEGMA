#include <PLEGMA_Fmunu.h>
#include <PLEGMA_Gauge.h>
#include <kernels/PLEGMA_fmunu_utils.cuh>
using namespace plegma;
//---------------------------//
// class PLEGMA_Fmunu //
//---------------------------//

template<typename Float>
PLEGMA_Fmunu<Float>::PLEGMA_Fmunu(ALLOCATION_FLAG alloc_flag, GHOST_FLAG ghost_flag): 
  PLEGMA_Field<Float>(alloc_flag, FMUNU, ghost_flag){ ; }

template<typename Float>
void PLEGMA_Fmunu<Float>::compute_leaves(PLEGMA_Gauge<Float> &gauge){
  if(!this->IsAllocDevice())PLEGMA_error("Computation of Fmunu is only supported for GPUs");
  if(gauge.Ghost_flag() < FIRST_CORNER) PLEGMA_error("Cannot compute clover leaves unless gauge field can communicate corners");
  gauge.communicateGhost(); // this communicates also the corners we need to do the clover leaves
  clover_leaves_k(*this,gauge);
  std::complex<Float> coeff = {0.,-0.125};
  this->cscale(coeff);
}

template class PLEGMA_Fmunu<float>;
template class PLEGMA_Fmunu<double>;
