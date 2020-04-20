#include <PLEGMA_kernel_complex.cuh>

#ifndef PLEGMA_GAMMAS_SCATT_CUH
#define PLEGMA_GAMMAS_SCATT_CUH
namespace plegma{

  const __device__ float gamma_scatt[35][4][2] =
    {{{1,0},{1,0},{1,0},{1,0}},     // 1
     {{0,1},{0,1},{0,-1},{0,-1}},   // g1
     {{1,0},{-1,0},{-1,0},{1,0}},   // g2
     {{0,1},{0,-1},{0,-1},{0,1}},   // g3
     {{1,0},{1,0},{-1,0},{-1,0}},   // g4
     {{1,0},{1,0},{1,0},{1,0}},     // g5
     {{-1,0},{1,0},{1,0},{-1,0}},   // Cgx
     {{0,-1},{0,-1},{0,1},{0,1}},   // Cgy
     {{1,0},{1,0},{-1,0},{-1,0}},   // Cgz
     {{-1,0},{1,0},{-1,0},{1,0}},   // Cgxgt
     {{0,-1},{0,-1},{0,-1},{0,-1}}, // Cgygt
     {{1,0},{1,0},{1,0},{1,0}},     // Cgzgt
     {{-1,0},{1,0},{-1,0},{1,0}},   // Cgxgtg5
     {{0,-1},{0,-1},{0,-1},{0,-1}}, // Cgygtg5
     {{1,0},{1,0},{1,0},{1,0}},      // Cgzgtg5
     {{0,-1},{0,-1},{0,1},{0,1}},    //g5g1
     {{0,1},{0,1},{0,-1},{0,-1}},    //g1g5
     {{-1,0},{1,0},{1,0},{-1,0}},    //g5g2
     {{1,0},{-1,0},{-1,0},{1,0}},    //g2g5
     {{0,-1},{0,1},{0,1},{0,-1}},    //g5g3
     {{0,1},{0,-1},{0,-1},{0,1}},    //g3g5
     {{-1,0},{-1,0},{1,0},{1,0}},    //g5g4
     {{1,0},{1,0},{-1,0},{-1,0}},    //g4g5
     {{1,0},{-1,0},{-1,0},{1,0}},    //g5Cgx
     {{-1,0},{1,0},{1,0},{-1,0}},    //Cgxg5
     {{0,1},{0,1},{0,-1},{0,-1}},    //g5Cgy
     {{0,-1},{0,-1},{0,1},{0,1}},    //Cgyg5
     {{-1,0},{-1,0},{1,0},{1,0}},    //g5Cgz
     {{1,0},{1,0},{-1,0},{-1,0}},    //Cgzg5
     {{0,-1},{0,1},{0,-1},{0,1}},    //C
     {{0,-1},{0,1},{0,-1},{0,1}},    //Cg5
     {{0,1},{0,-1},{0,-1},{0,1}},    //Cgt
     {{0,-1},{0,1},{0,1},{0,-1}},    //Cg5gt
     {{0,1},{0,-1},{0,-1},{0,1}},    //Cgtg5
     {{0,-1},{0,1},{0,1},{0,-1}},    //Cg5gtg5
    };
  const __device__ short int gammaInd_scatt[35][4][2] =
    {{{0,0},{1,1},{2,2},{3,3}},     //1
     {{0,3},{1,2},{2,1},{3,0}},     // g1
     {{0,3},{1,2},{2,1},{3,0}},     // g2
     {{0,2},{1,3},{2,0},{3,1}},     // g3
     {{0,0},{1,1},{2,2},{3,3}},     // g4
     {{0,2},{1,3},{2,0},{3,1}},     // g5
     {{0,0},{1,1},{2,2},{3,3}},     // Cgx
     {{0,0},{1,1},{2,2},{3,3}},     // Cgy
     {{0,1},{1,0},{2,3},{3,2}},     // Cgz
     {{0,2},{1,3},{2,0},{3,1}},     // Cgxgt
     {{0,0},{1,1},{2,2},{3,3}},     // Cgygt
     {{0,2},{1,3},{2,0},{3,1}},     // Cgzgt
     {{0,2},{1,3},{2,0},{3,1}},     // Cgxgtg5
     {{0,2},{1,3},{2,0},{3,1}},     // Cgygtg5
     {{0,3},{1,2},{2,1},{3,0}},     // Cgzgtg5
     {{0,1},{1,0},{2,3},{3,2}},     // g5g1
     {{0,1},{1,0},{2,3},{3,2}},     // g1g5
     {{0,1},{1,0},{2,3},{3,2}},     // g5g2
     {{0,1},{1,0},{2,3},{3,2}},     // g2g5
     {{0,0},{1,1},{2,2},{3,3}},     // g5g3
     {{0,0},{1,1},{2,2},{3,3}},     // g3g5
     {{0,2},{1,3},{2,0},{3,1}},     // g5g4
     {{0,2},{1,3},{2,0},{3,1}},     // g4g5
     {{0,2},{1,3},{2,0},{3,1}},     // g5Cgx
     {{0,2},{1,3},{2,0},{3,1}},     // Cgxg5
     {{0,2},{1,3},{2,0},{3,1}},     // g5Cgy
     {{0,2},{1,3},{2,0},{3,1}},     // Cgyg5
     {{0,3},{1,2},{2,1},{3,0}},     // g5Cgz
     {{0,3},{1,2},{2,1},{3,0}},     // Cgzg5
     {{0,3},{1,2},{2,1},{3,0}},     // C
     {{0,1},{1,0},{2,3},{3,2}},     // Cg5
     {{0,3},{1,2},{2,1},{3,0}},     // Cgt
     {{0,1},{1,0},{2,3},{3,2}},     // Cg5gt
     {{0,1},{1,0},{2,3},{3,2}},     // Cgtg5
     {{0,3},{1,2},{2,1},{3,0}},     // Cg5gtg5
    };

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
    dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
    dim3 gridDim( (HGC_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
    switch(LR){
      case(LEFT):
        apply_gamma_scatt_vector_kernel<LEFT><<<gridDim,blockDim>>>(inOut, r);
        break;
      case(RIGHT):
        apply_gamma_scatt_vector_kernel<RIGHT><<<gridDim,blockDim>>>(inOut, r);
        break;
    }
    checkCudaError();
  }

}

#endif
