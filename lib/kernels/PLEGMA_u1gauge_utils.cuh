#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_kernel_tuner.cuh"
using namespace plegma;
using namespace quda;

template<typename Float>
static __global__ void constField_kernel(u1gauge2<Float> G, int mu, int nu, Float exparg, int xnu_0){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> gg;
  if (sid >= G.volume()) return;
  int x[N_DIMS] = GET_ID(sid);
#pragma unroll
  for(int i =0; i < N_DIMS; i++){
    if(i==mu){
      Float val;
      int xmx0=DGC_procPosition[nu]*DGC_localL[nu]+x[nu] - xnu_0;
      val =   exparg * xmx0 ;
      gg.x = cos(val); gg.y = sin(val);
    }
    else{
      gg=1.;
    }
    G.set(i,sid,gg);
  }
}

template<typename Float>
void constFieldU1(u1gauge2<Float> G, int mu, int nu, Float exparg, int xnu_0){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (G.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  constField_kernel<<<gridDim,blockDim>>>(G,mu,nu,exparg,xnu_0);
  checkQudaError();
}
