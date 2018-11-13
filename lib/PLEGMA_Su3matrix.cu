#include <PLEGMA_Su3matrix.h>
#include <PLEGMA_Gauge.h>
using namespace plegma;

//--------------------------//
// class PLEGMA_Su3matrix //
//--------------------------//

template<typename Float>
PLEGMA_Su3matrix<Float>::PLEGMA_Su3matrix(ALLOCATION_FLAG alloc_flag): 
  PLEGMA_Field<Float>(alloc_flag, SU3FIELD){ ; }

template<typename Float>
void PLEGMA_Su3matrix<Float>::absorbDir(PLEGMA_Gauge<Float> &u,int dir){
  cudaMemcpy(PLEGMA_Field<Float>::d_elem, u.D_elem()+dir*(PLEGMA_Field<Float>::field_length)*(PLEGMA_Field<Float>::total_length)*2,
  	     PLEGMA_Field<Float>::bytes_total_length, cudaMemcpyDeviceToDevice);
  checkCudaError();
}

template class PLEGMA_Su3matrix<float>;
template class PLEGMA_Su3matrix<double>;
