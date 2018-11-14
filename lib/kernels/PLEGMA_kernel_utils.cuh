#include <PLEGMA.h>
#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <PLEGMA_kernel_extern.cuh>
#include <PLEGMA_kernel_complex.cuh>
#include <PLEGMA_kernel_getSet.cuh>

#ifndef PLEGMA_KERNEL_UTILS_CUH
#define PLEGMA_KERNEL_UTILS_CUH

#define THREADS_PER_BLOCK 64
//#define TIMING_REPORT

namespace plegma {

  template<typename FloatA, typename FloatB>
  __inline__ __device__ void Gdag(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = b[j][i].conj();
    }
  }


  template<typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void mul_G_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = 0.;
      #pragma unroll
      for(int k=0; k<N_COLS; k++) {
        a[i][j] = a[i][j] + b[i][k]*c[k][j];
      }
    }
  }

  template<typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void mul_G_Gdag(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = 0.;
      #pragma unroll
      for(int k=0; k<N_COLS; k++) {
        a[i][j] = a[i][j] + b[i][k]*c[j][k].conj();
      }
    }
  }

  
  template<typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void mul_Gdag_Gdag(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
    #pragma unroll
    for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = 0.;
      #pragma unroll
      for(int k=0; k<N_COLS; k++) {
        Float2<FloatA> tmp = c[j][k]*b[k][i];
        a[i][j].x += tmp.x;
        a[i][j].y -= tmp.y;
      }
    }
  }

  template<typename FloatA, typename FloatB>
  __inline__ __device__ FloatA real_trace_mul_G_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS]){
    FloatA tr=0.;
    #pragma unroll
    for(int i=0; i<N_COLS; i++){
      #pragma unroll
      for(int j=0; j<N_COLS; j++) {
	tr+=(a[i][j]*b[j][i]).x;
      }
    }
    return tr;
  }

  template<typename FloatOutV, typename FloatG, typename FloatInV>
  __inline__ __device__ void mul_G_V(Float2<FloatOutV> outV[N_SPINS][N_COLS],
				     Float2<FloatG> G[N_COLS][N_COLS],
				     Float2<FloatInV> inV[N_SPINS][N_COLS]){
     #pragma unroll
     for(int mu=0; mu<N_SPINS; mu++)
       #pragma unroll
       for(int j=0; j<N_COLS; j++) {
	 outV[mu][j] = 0.;
         #pragma unroll
	 for(int k=0; k<N_COLS; k++) {
	   outV[mu][j] = outV[mu][j] + G[j][k]*inV[mu][k];
	 }
       }
  }

  template<typename FloatOutV, typename FloatG, typename FloatInV>
  __inline__ __device__ void mul_Gdag_V(Float2<FloatOutV> outV[N_SPINS][N_COLS],
					Float2<FloatG> G[N_COLS][N_COLS],
					Float2<FloatInV> inV[N_SPINS][N_COLS]){
     #pragma unroll
     for(int mu=0; mu<N_SPINS; mu++)
       #pragma unroll
       for(int j=0; j<N_COLS; j++) {
	 outV[mu][j] = 0.;
         #pragma unroll
	 for(int k=0; k<N_COLS; k++) {
	   outV[mu][j] = outV[mu][j] + conj(G[k][j])*inV[mu][k];
	 }
       }
  }
}
#endif
