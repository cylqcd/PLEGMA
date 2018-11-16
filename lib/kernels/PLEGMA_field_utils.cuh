#include <cublas_v2.h>
#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename FloatOut,typename FloatIn1, typename FloatIn2, typename Float>
static __global__ void xpby_kernel(FloatOut *z, FloatIn1 *x, FloatIn2 *y, Float beta, int length_field){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;
  generic2<FloatOut> Rz(z);
  generic2<FloatIn1> Ry(y);
  generic2<FloatIn2> Rx(x);
  for(int i = 0 ; i < length_field ; i++)
    Rz.set(Rx.get(i,sid) + beta*Ry.get(i,sid));
}

template<typename Float>
static void xpby(PLEGMA_Field<Float> &Fz, PLEGMA_Field<Float> &Fx, PLEGMA_Field<Float> &Fy, Float beta){
  if(Fz.Field_length() != Fx.Field_length()) errorQuda("Error input, output fields do not match");
  if(Fz.Field_length() != Fy.Field_length()) errorQuda("Error input, output fields do not match");
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  xpby_kernel<<<gridDim,blockDim>>>(Fz.D_elem(), Fx.D_elem(), Fy.D_elem(),beta,Fz.Field_length());
  checkCudaError();
}
