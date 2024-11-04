#include <PLEGMA_kernel_complex.cuh>

#ifndef PLEGMA_GAMMAS_SCATT_CUH
#define PLEGMA_GAMMAS_SCATT_CUH
namespace plegma{
  template<LEFTRIGHT LF,typename Float>
  __inline__ __device__ void gamma_scattV(Float2<Float> vout[N_SPINS][N_COLS], Float2<Float>vin[N_SPINS][N_COLS], short int r){
    const Float2<float> (*gamma2)[4];
    gamma2=(Float2<float> (*)[4]) plegma::gamma_scatt;
#pragma unroll
    for(int nz = 0; nz < N_SPINS; nz++){
      int mu = (LF == LEFT)? gammaInd_scatt[r][nz][0] : gammaInd_scatt[r][nz][1];
      int nu = (LF == LEFT)? gammaInd_scatt[r][nz][1] : gammaInd_scatt[r][nz][0];
#pragma unroll
      for(int c1 = 0; c1 < N_COLS; c1++)
        vout[mu][c1] = vin[nu][c1] * gamma2[r][nz] ;
    }
  }
  template<LEFTRIGHT LF,typename Float>
  static __global__ void apply_gamma_scatt_vector_kernel(vector2<Float> vec, GAMMAS_SCATT r){
    int sid = blockIdx.x*blockDim.x + threadIdx.x;
    Float2<Float> Sin[N_SPINS][N_COLS];
    Float2<Float> Sout[N_SPINS][N_COLS];
    if (sid >= DGC_localVolume) return;
    vec.get(Sin,sid);
    gamma_scattV<LF>(Sout,Sin,r);
    vec.set(Sout,sid);
  }

  template<typename Float>
  static void apply_gamma_scatt_vector(LEFTRIGHT LR, vector2<Float> inOut, GAMMAS_SCATT r){
    dim3 blockDim( inOut.volume() , 1, 1);
    dim3 gridDim( ( + blockDim.x -1)/blockDim.x , 1 , 1);
    switch(LR){
      case(LEFT):
        apply_gamma_scatt_vector_kernel<LEFT><<<gridDim,blockDim>>>(inOut, r);
        break;
      case(RIGHT):
        apply_gamma_scatt_vector_kernel<RIGHT><<<gridDim,blockDim>>>(inOut, r);
        break;
    }
    checkQudaError();
  }

}

#endif
