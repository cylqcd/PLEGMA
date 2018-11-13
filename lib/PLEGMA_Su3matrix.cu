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
void PLEGMA_Su3matrix<Float>::absorbDir_device(PLEGMA_Gauge<Float> &u,int dir){
  cudaMemcpy(this->d_elem, u.D_elem()+dir*(this->field_length)*(this->total_length)*2,
  	     this->bytes_total_length, cudaMemcpyDeviceToDevice);
  checkCudaError();
}

template<typename Float>
void PLEGMA_Su3matrix<Float>::absorbDir_host(PLEGMA_Gauge<Float> &u,int dir){
  memcpy(this->h_elem, u.H_elem()+dir*(this->field_length)*(this->total_length)*2,
	 this->bytes_total_length);
}

template class PLEGMA_Su3matrix<float>;
template class PLEGMA_Su3matrix<double>;
