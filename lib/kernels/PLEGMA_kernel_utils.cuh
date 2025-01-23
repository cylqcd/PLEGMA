#include <errno.h>
#include <mpi.h>  
#include <limits>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <typeinfo>
#include <PLEGMA_kernel_complex.cuh>
#include <PLEGMA_kernel_getSet.cuh>
#include <PLEGMA_kernel_tuner.cuh>
#include <PLEGMA_gammas.cuh>

#ifndef PLEGMA_KERNEL_UTILS_CUH
#define PLEGMA_KERNEL_UTILS_CUH

#define THREADS_PER_BLOCK 64
//#define TIMING_REPORT



/* 
 * From https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
 * Note that any atomic operation can be implemented based on atomicCAS() (Compare And Swap). 
 * For example, atomicAdd() for double-precision floating-point numbers is not available 
 * on devices with compute capability lower than 6.0 but it can be implemented as follows:
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
static __inline__  __device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
		    __double_as_longlong(val +
					 __longlong_as_double(assumed)));
    
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);
  
  return __longlong_as_double(old);
}
#endif

using namespace plegma;

namespace plegma {

  static const __device__ int eps[6][3]= {{0,1,2},
					  {2,0,1},
					  {1,2,0},
					  {2,1,0},
					  {0,2,1},
					  {1,0,2}};
    
  static const __device__ int sgn_eps[6]= { +1,+1,+1,-1,-1,-1 };
  enum TMROT {NOROT, TMP, TMM};

  template<bool isTransMatrix,typename Float>
  __inline__ __device__ Float2<Float> trace_gamma_S(int opId, TMROT trot, Float2<Float> R[N_SPINS][N_SPINS]){
      const Float2<float> (*g)[4];
      const short int (*gIn)[4][2];
      if(trot == TMP){
	g = (Float2<float> (*)[4]) gammaTmP;
	gIn = gammaIndTmP; 
      }else if (trot == TMM){
	g = (Float2<float> (*)[4]) gammaTmM;
	gIn = gammaIndTmM;     
      }
      else{
	g = (Float2<float> (*)[4]) gamma;
	gIn = gammaInd;
      }
      Float2<Float> accum = 0.;
#pragma unroll
      for(int nz = 0 ; nz < N_SPINS ; nz++){
	int mu = gIn[opId][nz][0];
        int nu = gIn[opId][nz][1];
        Float2<Float> val = g[opId][nz];
        accum += isTransMatrix ? val*R[mu][nu] : val*R[nu][mu];
      }
      return accum;
  }

  template<bool isTransMatrix,typename Float>
  __inline__ __device__ Float2<Float> trace_spin_color_V_gamma_V(int opId, TMROT trot, Float2<Float> vec1[N_SPINS][N_COLS], Float2<Float> vec2[N_SPINS][N_COLS]){
      const Float2<float> (*g)[4];
      const short int (*gIn)[4][2];
      if(trot == TMP){
        g = (Float2<float> (*)[4]) gammaTmP;
        gIn = gammaIndTmP;
      }else if (trot == TMM){
        g = (Float2<float> (*)[4]) gammaTmM;
        gIn = gammaIndTmM;
      }
      else{
        g = (Float2<float> (*)[4]) gamma;
        gIn = gammaInd;
      }
      Float2<Float> accum = 0.;
#pragma unroll
      for(int nz = 0 ; nz < N_SPINS ; nz++){
        int mu = gIn[opId][nz][0];
        int nu = gIn[opId][nz][1];
        Float2<Float> val = g[opId][nz];
        #pragma unroll 
        for (int color=0; color<N_COLS; ++color){
          accum += isTransMatrix ? val*vec1[mu][color]*vec2[nu][color] : val*vec1[nu][color]*vec2[mu][color];
	}
      }
      return accum;
  }

  template<bool isTransMatrix,typename Float>
  __inline__ __device__ Float2<Float> trace_spin_V_gamma_V(int opId, TMROT trot, Float2<Float> vec1[N_SPINS], Float2<Float> vec2[N_SPINS]){
      const Float2<float> (*g)[4];
      const short int (*gIn)[4][2];
      if(trot == TMP){
        g = (Float2<float> (*)[4]) gammaTmP;
        gIn = gammaIndTmP;
      }else if (trot == TMM){
        g = (Float2<float> (*)[4]) gammaTmM;
        gIn = gammaIndTmM;
      }
      else{
        g = (Float2<float> (*)[4]) gamma;
        gIn = gammaInd;
      }
      Float2<Float> accum = 0.;
#pragma unroll
      for(int nz = 0 ; nz < N_SPINS ; nz++){
        int mu = gIn[opId][nz][0];
        int nu = gIn[opId][nz][1];
        Float2<Float> val = g[opId][nz];
        accum += isTransMatrix ? val*vec1[mu]*vec2[nu] : val*vec1[nu]*vec2[mu];
      }
      return accum;
  }


  
  template<LEFTRIGHT LF,typename Float>
  __inline__ __device__ void gammaV(Float2<Float> vout[N_SPINS][N_COLS], Float2<Float>vin[N_SPINS][N_COLS], short int r){
    const Float2<float> (*gamma2)[4];
    gamma2=(Float2<float> (*)[4]) plegma::gamma;
#pragma unroll
    for(int nz = 0; nz < N_SPINS; nz++){
      int mu = (LF == LEFT)? gammaInd[r][nz][0] : gammaInd[r][nz][1];
      int nu = (LF == LEFT)? gammaInd[r][nz][1] : gammaInd[r][nz][0];
#pragma unroll
      for(int c1 = 0; c1 < N_COLS; c1++)
	vout[mu][c1] = vin[nu][c1] * gamma2[r][nz] ;
    }
  }


  template<typename Float>
  __inline__ __device__ void U_uk_ch_g5g4(Float2<Float> vout[N_SPINS][N_COLS], Float2<Float>vin[N_SPINS][N_COLS]){
    Float nrm=1./sqrt(2.);
#pragma unroll
    for(int c1 = 0; c1 < N_COLS; c1++){
      vout[0][c1] = nrm * (vin[2][c1] - vin[0][c1]);
      vout[1][c1] = nrm * (vin[3][c1] - vin[1][c1]);
      vout[2][c1] = nrm * (vin[2][c1] + vin[0][c1]);
      vout[3][c1] = nrm * (vin[3][c1] + vin[1][c1]);
    }
  }


  template<typename Float>
  __inline__ __device__ void U_uk_ch_etmc(Float2<Float> vout[N_SPINS][N_COLS], Float2<Float>vin[N_SPINS][N_COLS]){
    Float nrm=1./sqrt(2.);
#pragma unroll
    for(int c1 = 0; c1 < N_COLS; c1++){
      vout[0][c1] = nrm * (vin[0][c1] + vin[2][c1]);
      vout[1][c1] = nrm * (vin[1][c1] + vin[3][c1]);
      vout[2][c1] = nrm * (vin[0][c1] - vin[2][c1]);
      vout[3][c1] = nrm * (vin[1][c1] - vin[3][c1]);
    }
  }


  
  template<LEFTRIGHT LF,typename Float>
  __inline__ __device__ void gammaProp(Float2<Float> pout[N_SPINS][N_SPINS][N_COLS][N_COLS],
				       Float2<Float> pin[N_SPINS][N_SPINS][N_COLS][N_COLS], short int r){
    const Float2<float> (*gamma2)[4];
    gamma2=(Float2<float> (*)[4]) plegma::gamma;
#pragma unroll
    for(int nu = 0 ; nu < N_SPINS; nu++)
#pragma unroll
    for(int nz = 0; nz < N_SPINS; nz++){
      int mu = (LF == LEFT)? gammaInd[r][nz][0] : gammaInd[r][nz][1];
      int rho = (LF == LEFT)? gammaInd[r][nz][1] : gammaInd[r][nz][0];
#pragma unroll
      for(int c1 = 0; c1 < N_COLS; c1++)
	for(int c2 = 0; c2 < N_COLS; c2++){
	  Float2<Float> p;
	  p = (LF == LEFT)? pin[rho][nu][c1][c2]: pin[nu][rho][c1][c2];
	  if (LF == LEFT) pout[mu][nu][c1][c2] = p*gamma2[r][nz];
	  else pout[nu][mu][c1][c2] = p*gamma2[r][nz] ;
	}
    }
  }

  template<typename Float>
  __inline__ __device__ Float2<Float> det(Float2<Float> a[N_COLS][N_COLS]){
  return a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] +a[0][0]*a[1][1]*a[2][2]
    - a[0][2]*a[1][1]*a[2][0] - a[0][0]*a[1][2]*a[2][1] - a[0][1]*a[1][0]*a[2][2];
  }
  
  template<typename FloatR, typename FloatU>
  __inline__ __device__ FloatR real_trace(Float2<FloatU> a[N_COLS][N_COLS]){
    FloatR r = a[0][0].x+a[1][1].x+a[2][2].x;
    return r;
  }
  
  template<typename FloatR, typename FloatU>
  __inline__ __device__ Float2<FloatR> trace(Float2<FloatU> a[N_COLS][N_COLS]){
    Float2<FloatR> r = a[0][0]+a[1][1]+a[2][2];
    return r;
  }

  template<typename Float>
  __inline__ __device__ void enforce_herm(Float2<Float> H[3][3]){
#pragma unroll
  for(int i=0; i<N_COLS-1; i++)
#pragma unroll
    for(int j=i+1; j<N_COLS; j++){
      H[i][j] = (H[i][j] + conj(H[j][i]))/((Float) 2.);
      H[j][i] = conj(H[i][j]);
    }
  }

  template<typename FloatA, typename FloatB>
  __inline__ __device__ void Gdag(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = conj(b[j][i]);
    }
  }

  template<typename FloatA, typename FloatB>
  __inline__ __device__ void Gtrans(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = b[j][i];
    }
  }

  template<typename FloatA>
  __inline__ __device__ void Gtrans(Float2<FloatA> a[N_COLS][N_COLS]){
    Float2<FloatA> E;
#pragma unroll
  for(int i=0; i<N_COLS-1; i++)
    #pragma unroll
    for(int j=i+1; j<N_COLS; j++) {
      E=a[i][j];
      a[i][j] = a[j][i];
      a[j][i] = E;
    }
  }


  template<typename FloatA>
  __inline__ __device__ void Gconj(Float2<FloatA> a[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j].conj();
    }
  }

  template<typename FloatA>
  __inline__ __device__ void Gdag(Float2<FloatA> a[N_COLS][N_COLS]){
    Gtrans(a);
    Gconj(a);
  }
  
  template<typename T, typename FloatG>
  __inline__ __device__ void scaleG(Float2<FloatG> a[N_COLS][N_COLS], T w){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = w*a[i][j];
    }
  }

  template<typename T, typename FloatG>
  __inline__ __device__ void G_plus_aG(Float2<FloatG> a[N_COLS][N_COLS], Float2<FloatG> b[N_COLS][N_COLS], T w){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = a[i][j] + w*b[i][j];
    }
  }

  template<typename T, typename FloatG>
  __inline__ __device__ void G_plus_aG(Float2<FloatG> a[N_COLS][N_COLS], Float2<FloatG> b[N_COLS][N_COLS], Float2<FloatG> c[N_COLS][N_COLS], T w){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = b[i][j] + w*c[i][j];
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

  template<typename Float,typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ Float2<Float> trace_mul_G_G_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
    Float2<Float> res = 0.;
#pragma unroll
    for(int i=0; i<N_COLS; i++)
#pragma unroll
      for(int j=0; j<N_COLS; j++) {
#pragma unroll
	for(int k=0; k<N_COLS; k++) {
	  res += a[i][j]*b[j][k]*c[k][i];
	}
      }
    return res;
  }


  template<typename Float,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ Float2<Float> trace_mul_G_G_G_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS],
							Float2<FloatC> c[N_COLS][N_COLS], Float2<FloatD> d[N_COLS][N_COLS]){
    Float2<FloatA> tmp1[N_COLS][N_COLS], tmp2[N_COLS][N_COLS];
    Float2<FloatA> res=0;
    
#pragma unroll
    for(int i = 0; i < N_COLS; i++)
#pragma unroll
      for(int j = 0; j < N_COLS; j++){
	tmp1[i][j]=0.; tmp2[i][j]=0.;
#pragma unroll
	for(int k = 0; k < N_COLS; k++){
	  tmp1[i][j] += a[i][k]*b[k][j];
	  tmp2[i][j] += c[i][k]*d[k][j];
	}
      }

#pragma unroll
    for(int i = 0; i < N_COLS; i++)
#pragma unroll
      for(int j = 0; j < N_COLS; j++){
	res += tmp1[i][j]*tmp2[j][i];
      }
    return res;
  }

  template<typename FloatOut, typename FloatG, typename FloatIn, ACCUM_TYPE accum=ACC_ZERO>
    __inline__ __device__ void mul_G(Float2<FloatOut> out[N_SPINS][N_SPINS][N_COLS][N_COLS],
                                       Float2<FloatG> G[N_COLS][N_COLS],
                                       Float2<FloatIn> in[N_SPINS][N_SPINS][N_COLS][N_COLS]){
    if(accum==ACC_ZERO || accum == ZERO_PLUS || accum == ZERO_MINUS)
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++)
        #pragma unroll
        for(int nu=0; nu<N_SPINS; nu++)
          #pragma unroll
          for(int i=0; i<N_COLS; i++)
            #pragma unroll
            for(int j=0; j<N_COLS; j++)
              out[mu][nu][i][j] = 0.;

    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++)
      #pragma unroll
      for(int nu=0; nu<N_SPINS; nu++)
        #pragma unroll
        for(int j=0; j<N_COLS; j++)
          #pragma unroll
          for(int i=0; i<N_COLS; i++)
            #pragma unroll
            for(int k=0; k<N_COLS; k++) {
              if(accum==ACC_MINUS || accum == ZERO_MINUS) out[mu][nu][j][i] -= G[j][k]*in[mu][nu][k][i];
              else out[mu][nu][j][i] += G[j][k]*in[mu][nu][k][i];
            }
  }


  template<typename FloatOut, typename FloatG, typename FloatIn, ACCUM_TYPE accum=ACC_ZERO>
  __inline__ __device__ void mul_Gdag(Float2<FloatOut> out[N_SPINS][N_SPINS][N_COLS][N_COLS],
                                      Float2<FloatG> G[N_COLS][N_COLS],
                                      Float2<FloatIn> in[N_SPINS][N_SPINS][N_COLS][N_COLS]){
    if(accum==ACC_ZERO || accum == ZERO_PLUS || accum == ZERO_MINUS)
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++)
        #pragma unroll
        for(int nu=0; nu<N_SPINS; nu++)
          #pragma unroll
          for(int i=0; i<N_COLS; i++)
            #pragma unroll
            for(int j=0; j<N_COLS; j++)
              out[mu][nu][i][j] = 0.;

    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++)
      #pragma unroll
      for(int nu=0; nu<N_SPINS; nu++)
        #pragma unroll
        for(int j=0; j<N_COLS; j++)
          #pragma unroll
          for(int i=0; i<N_COLS; i++)
            #pragma unroll
            for(int k=0; k<N_COLS; k++) {
              if(accum==ACC_MINUS || accum == ZERO_MINUS) out[mu][nu][j][i] -= conj(G[k][j])*in[mu][nu][k][i];
              else out[mu][nu][j][i] += conj(G[k][j])*in[mu][nu][k][i];
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
        a[i][j] = a[i][j] + b[i][k]*conj(c[j][k]);
      }
    }
  }

  template<typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void mul_Gdag_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++)
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j] = 0.;
      #pragma unroll
      for(int k=0; k<N_COLS; k++) {
        a[i][j] = a[i][j] + conj(b[k][i])*c[k][j];
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

  template<bool isDagB, bool isDagC, typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void mul_G_G(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS], Float2<FloatC> c[N_COLS][N_COLS]){
    if(isDagB and isDagC)
      mul_Gdag_Gdag(a,b,c);
    else if(isDagB)
      mul_Gdag_G(a,b,c);
    else if(isDagC)
      mul_G_Gdag(a,b,c);
    else
      mul_G_G(a,b,c);
  }

  template<bool isLeftTrans, ACCUM_TYPE aty,typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void partial_trace_mul_Prop_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
							 Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
							 Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS]){
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
	if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS) {
	  A[mu][nu].x=0.; A[mu][nu].y=0.;
	}
#pragma unroll
	for(int rho = 0 ; rho < N_SPINS; rho++)
#pragma unroll
	  for(int a = 0; a < N_COLS; a++)
#pragma unroll
	    for(int b = 0; b < N_COLS; b++){
	      if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
		if(isLeftTrans) A[mu][nu] +=  B[mu][rho][b][a] * C[nu][rho][b][a];
		else A[mu][nu] +=  B[rho][mu][a][b] * C[nu][rho][b][a];
	      }
	      else{
		if(isLeftTrans) A[mu][nu] -=  B[mu][rho][b][a] * C[nu][rho][b][a];
		else A[mu][nu] -=  B[rho][mu][a][b] * C[nu][rho][b][a];		
	      }
	    }
	
      }
  }

  template<bool isLeftTrans, ACCUM_TYPE aty,typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void partial_trace_mul_Prop_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                         Float2<FloatB> B[N_SPINS][N_COLS],
                                                         Float2<FloatC> C[N_SPINS][N_COLS]){
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
        if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS) {
          A[mu][nu].x=0.; A[mu][nu].y=0.;
        }
#pragma unroll
        for(int a = 0; a < N_COLS; a++)
          if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
            if(isLeftTrans) A[mu][nu] +=  B[mu][a] * C[nu][a];
              else A[mu][nu] +=  B[mu][a] * C[nu][a];
	    }
            else{
              if(isLeftTrans) A[mu][nu] -=  B[mu][a] * C[nu][a];
              else A[mu][nu] -=  B[mu][a] * C[nu][a];
            }
            

      }
  }


  template<bool isLeftTrans, ACCUM_TYPE aty,typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void open_mul_Prop_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
						Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
						Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
						int s1, int s2, int c1, int c2){
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
	if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS) {
	  A[mu][nu].x=0.; A[mu][nu].y=0.;
	}
#pragma unroll
	for(int b = 0; b < N_COLS; b++){
	  if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
	    if(isLeftTrans) A[mu][nu] +=  B[mu][s1][b][c1] * C[nu][s2][b][c2];
	    else A[mu][nu] +=  B[s1][mu][c1][b] * C[nu][s2][b][c2];
	  }
	  else{
	    if(isLeftTrans) A[mu][nu] -=  B[mu][s1][b][c1] * C[nu][s2][b][c2];
	    else A[mu][nu] -=  B[s1][mu][c1][b] * C[nu][s2][b][c2];		
	  }
	}
	
      }
  }

  template<bool isLeftTrans, ACCUM_TYPE aty,typename FloatA, typename FloatB, typename FloatC>
  __inline__ __device__ void open_mul_Prop_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                Float2<FloatB> B[N_SPINS][N_COLS],
                                                Float2<FloatC> C[N_SPINS][N_COLS],
                                                int s1, int s2, int c1, int c2){
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
        if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS) {
          A[mu][nu].x=0.; A[mu][nu].y=0.;
        }
#pragma unroll
        for(int b = 0; b < N_COLS; b++){
          if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
            if(isLeftTrans) A[mu][nu] +=  B[mu][b] * C[nu][b];
            else A[mu][nu] +=  B[mu][b] * C[nu][b];
          }
          else{
            if(isLeftTrans) A[mu][nu] -=  B[mu][b] * C[nu][b];
            else A[mu][nu] -=  B[mu][b] * C[nu][b];
          }
        }

      }
  }
  
  template<bool isLeftTrans, ACCUM_TYPE aty, bool isGdag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void open_mul_Prop_G_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
						  Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
						  Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
						  Float2<FloatD> D[N_COLS][N_COLS],
						  int s1,int s2,int c1, int c2){
    if(isGdag) Gdag(D);
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
	if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS) {
	  A[mu][nu].x=0.; A[mu][nu].y=0.;
	}
#pragma unroll
	for(int b = 0; b < N_COLS; b++)
#pragma unroll
	  for(int c = 0; c < N_COLS; c++){
	    if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
	      if(isLeftTrans) A[mu][nu] +=  B[mu][s1][b][c1] * D[b][c] * C[nu][s2][c][c2];
	      else A[mu][nu] += B[s1][mu][c1][b] * D[b][c] * C[nu][s2][c][c2];
	    }
	    else{
	      if(isLeftTrans) A[mu][nu] -=  B[mu][s1][b][c1] * D[b][c] * C[nu][s2][c][c2];
	      else A[mu][nu] -= B[s1][mu][c1][b] * D[b][c] * C[nu][s2][c][c2];
	    }
	  }
	
      }
    if(isGdag) Gdag(D);
  }
    
  template<bool isLeftTrans, ACCUM_TYPE aty, bool isGdag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
							   Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
							   Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
							   Float2<FloatD> D[N_COLS][N_COLS],
							   int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    if ((s1>=0) && (s2>=0) && (c1>=0) && (c2>=0)) {
      return open_mul_Prop_G_Prop<isLeftTrans,aty,isGdag>(A,B,C,D,s1,s2,c1,c2);
    }
    if(isGdag) Gdag(D);
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++)
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
	if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS){
	  A[mu][nu].x=0.; A[mu][nu].y=0.;
	}
#pragma unroll
	for(int rho = 0 ; rho < N_SPINS; rho++)
#pragma unroll
	  for(int a = 0; a < N_COLS; a++)
#pragma unroll
	    for(int b = 0; b < N_COLS; b++)
#pragma unroll
	      for(int c = 0; c < N_COLS; c++){
		if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
		  if(isLeftTrans) A[mu][nu] +=  B[mu][rho][b][a] * D[b][c] * C[nu][rho][c][a];
		  else A[mu][nu] += B[rho][mu][a][b] * D[b][c] * C[nu][rho][c][a];
		}
		else{
		  if(isLeftTrans) A[mu][nu] -=  B[mu][rho][b][a] * D[b][c] * C[nu][rho][c][a];
		  else A[mu][nu] -= B[rho][mu][a][b] * D[b][c] * C[nu][rho][c][a];
		}
	      }
	
      }
    if(isGdag) Gdag(D);
  }

  template<bool isLeftTrans, ACCUM_TYPE aty, bool isGdag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Vec_G_Vec(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                           Float2<FloatB> B[N_SPINS][N_COLS],
                                                           Float2<FloatC> C[N_SPINS][N_COLS],
                                                           Float2<FloatD> D[N_COLS][N_COLS],
                                                           int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    if(isGdag) Gdag(D);
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++){
#pragma unroll 
      for(int nu = 0 ; nu < N_SPINS; nu++){
        if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS){
           A[mu][nu].x=0.; A[mu][nu].y=0.;
        }
#pragma unroll
        for(int a = 0; a < N_COLS; a++)
#pragma unroll
          for(int b = 0; b < N_COLS; b++){
            if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
             if(isLeftTrans)  A[mu][nu] +=  B[mu][a] * D[b][a] * C[nu][b];
             else A[mu][nu] += B[mu][a] * D[a][b] * C[nu][b];
            }
            else{
             if(isLeftTrans) A[mu][nu] -=  B[mu][a] * D[b][a] * C[nu][b];
             else A[mu][nu] -= B[mu][a] * D[b][a] * C[nu][b];
            }
          }

      }
    }

    if(isGdag) Gdag(D);
    
  }

    template<bool isLeftTrans, ACCUM_TYPE aty, bool isGdag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                           Float2<FloatB> B[N_SPINS][N_COLS],
                                                           Float2<FloatC> C[N_SPINS][N_COLS],
                                                           Float2<FloatD> D[N_COLS][N_COLS],
							   int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    if(isGdag) Gdag(D);
#pragma unroll
    for(int mu = 0 ; mu < N_SPINS; mu++){
#pragma unroll
      for(int nu = 0 ; nu < N_SPINS; nu++){
        if(aty == ACC_ZERO || aty == ZERO_PLUS || aty == ZERO_MINUS){
           A[mu][nu].x=0.; A[mu][nu].y=0.;
        }
#pragma unroll
        for(int a = 0; a < N_COLS; a++)
#pragma unroll
          for(int b = 0; b < N_COLS; b++){
            if(aty == ACC_ZERO || aty == ACC_PLUS || aty == ZERO_PLUS){
             if(isLeftTrans)  A[mu][nu] +=  B[mu][a] * D[a][b] * C[nu][b];
             else A[mu][nu] += B[mu][a] * D[a][b] * C[nu][b];
            }
            else{
             if(isLeftTrans) A[mu][nu] -=  B[mu][a] * D[a][b] * C[nu][b];
             else A[mu][nu] -= B[mu][a] * D[a][b] * C[nu][b];
            }
          }

      }
    }

    if(isGdag) Gdag(D);

  }


  
  template<bool isLeftTrans, ACCUM_TYPE aty, bool isG1dag, bool isG2dag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G1_G2_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
							       Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
							       Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
							       Float2<FloatD> D1[N_COLS][N_COLS],
							       Float2<FloatD> D2[N_COLS][N_COLS],
							       int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    Float2<FloatD> D[N_COLS][N_COLS];
    mul_G_G<isG1dag,isG2dag>(D,D1,D2);
    partial_trace_mul_Prop_G_Prop<isLeftTrans,aty,false>(A,B,C,D,s1,s2,c1,c2);
  }

  template<bool isLeftTrans, ACCUM_TYPE aty, bool isG1dag, bool isG2dag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G1_G2_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                               Float2<FloatB> B[N_SPINS][N_COLS],
                                                               Float2<FloatC> C[N_SPINS][N_COLS],
                                                               Float2<FloatD> D1[N_COLS][N_COLS],
                                                               Float2<FloatD> D2[N_COLS][N_COLS],
                                                               int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    Float2<FloatD> D[N_COLS][N_COLS];
    mul_G_G<isG1dag,isG2dag>(D,D1,D2);
    partial_trace_mul_Prop_G_Prop<isLeftTrans,aty,false>(A,B,C,D,s1,s2,c1,c2);
  }


  template<bool isLeftTrans, ACCUM_TYPE aty, bool isG1dag, bool isG2dag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Vec_G1_G2_Vec(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                               Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
                                                               Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
                                                               Float2<FloatD> D1[N_COLS][N_COLS],
                                                               Float2<FloatD> D2[N_COLS][N_COLS]){
    Float2<FloatD> D[N_COLS][N_COLS];
    mul_G_G<isG1dag,isG2dag>(D,D1,D2);
    partial_trace_mul_Vec_G_Vec<isLeftTrans,aty,false>(A,B,C,D);
  }


  template<bool isLeftTrans, ACCUM_TYPE aty, bool isG1dag, bool isG2dag, bool isG3dag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G1_G2_G3_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
								  Float2<FloatB> B[N_SPINS][N_SPINS][N_COLS][N_COLS],
								  Float2<FloatC> C[N_SPINS][N_SPINS][N_COLS][N_COLS],
								  Float2<FloatD> D1[N_COLS][N_COLS],
								  Float2<FloatD> D2[N_COLS][N_COLS],
								  Float2<FloatD> D3[N_COLS][N_COLS],
								  int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    Float2<FloatD> D[N_COLS][N_COLS];
    mul_G_G<isG1dag,isG2dag>(D,D1,D2);
    partial_trace_mul_Prop_G1_G2_Prop<isLeftTrans,aty,false,isG3dag>(A,B,C,D,D3,s1,s2,c1,c2);
  }

  template<bool isLeftTrans, ACCUM_TYPE aty, bool isG1dag, bool isG2dag, bool isG3dag,typename FloatA, typename FloatB, typename FloatC, typename FloatD>
  __inline__ __device__ void partial_trace_mul_Prop_G1_G2_G3_Prop(Float2<FloatA> A[N_SPINS][N_SPINS],
                                                                  Float2<FloatB> B[N_SPINS][N_COLS],
                                                                  Float2<FloatC> C[N_SPINS][N_COLS],
                                                                  Float2<FloatD> D1[N_COLS][N_COLS],
                                                                  Float2<FloatD> D2[N_COLS][N_COLS],
                                                                  Float2<FloatD> D3[N_COLS][N_COLS],
                                                                  int s1=-1,int s2=-1,int c1=-1, int c2=-1){
    Float2<FloatD> D[N_COLS][N_COLS];
    mul_G_G<isG1dag,isG2dag>(D,D1,D2);
    partial_trace_mul_Prop_G1_G2_Prop<isLeftTrans,aty,false,isG3dag>(A,B,C,D,D3,s1,s2,c1,c2);
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

  template<typename FloatV>
  __inline__ __device__ bool isNotZeroV(Float2<FloatV> vec[N_SPINS][N_COLS]){
    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++)
      #pragma unroll
      for(int j=0; j<N_COLS; j++)
	if(not vec[mu][j].isZero()) return true;
    return false;
  }

  template<typename FloatOutV, typename FloatG, typename FloatInV, ACCUM_TYPE accum=ACC_ZERO>
    __inline__ __device__ void mul_G_V(Float2<FloatOutV> outV[N_SPINS][N_COLS],
				       Float2<FloatG> G[N_COLS][N_COLS],
				       Float2<FloatInV> inV[N_SPINS][N_COLS]){
      #pragma unroll
      for(int mu=0; mu<N_SPINS; mu++)
        #pragma unroll
        for(int j=0; j<N_COLS; j++) {
          if(accum==ACC_ZERO || accum == ZERO_PLUS || accum == ZERO_MINUS) outV[mu][j] = 0.;
          #pragma unroll
          for(int k=0; k<N_COLS; k++) {
            if(accum==ACC_MINUS || accum == ZERO_MINUS) outV[mu][j] -= G[j][k]*inV[mu][k];
            else outV[mu][j] += G[j][k]*inV[mu][k];
          }
        }
    }
  
  template<typename FloatOutV, typename FloatG, typename FloatInV, ACCUM_TYPE accum=ACC_ZERO>
  __inline__ __device__ void mul_Gdag_V(Float2<FloatOutV> outV[N_SPINS][N_COLS],
					Float2<FloatG> G[N_COLS][N_COLS],
					Float2<FloatInV> inV[N_SPINS][N_COLS]){
     #pragma unroll
     for(int mu=0; mu<N_SPINS; mu++)
       #pragma unroll
       for(int j=0; j<N_COLS; j++) {
	 if(accum==ACC_ZERO || accum == ZERO_PLUS || accum == ZERO_MINUS) outV[mu][j] = 0.;
         #pragma unroll
	 for(int k=0; k<N_COLS; k++) {
	   if(accum==ACC_MINUS || accum == ZERO_MINUS) outV[mu][j] -= conj(G[k][j])*inV[mu][k];
	   else outV[mu][j] += conj(G[k][j])*inV[mu][k];
	 }
       }
  }

  template<typename FloatOutV, typename FloatInV>
  __inline__ __device__ void Vdag_g5(Float2<FloatOutV> outV[N_SPINS][N_COLS], Float2<FloatInV> inV[N_SPINS][N_COLS]){
#pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++)
#pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
	outV[(mu+2)%4][c1] = conj(inV[mu][c1]);
  }
  
  template<typename T>
  __inline__ __device__ void reduce(T *shared_cache, const int n_comp){
    // synchronize threads to be sure that all have written their register trace to share memory
    int i = blockDim.x/2;
    int r = blockDim.x%2;
    while (i > 0){
      __syncthreads();
      if(threadIdx.x < i){
	for(int ip = 0 ; ip < n_comp ; ip++) {
	  shared_cache[ip*blockDim.x + threadIdx.x] = shared_cache[ip*blockDim.x + threadIdx.x] + shared_cache[ip*blockDim.x + threadIdx.x + i];
	  if(r==1 && threadIdx.x==i-1)
	    shared_cache[ip*blockDim.x + threadIdx.x] =  shared_cache[ip*blockDim.x + threadIdx.x] + shared_cache[ip*blockDim.x + threadIdx.x + i+1];
	}
      }
      r = i%2;
      i /= 2;
    }
  }

  template<typename Float>
  __inline__ __device__ void fourier_transform_3D( Float2<Float> *out, Float2<Float> *in,
						   Float2<Float> *shared_cache, int n_comp,
						   int sid3D, int sp[3], tex_mom_list &texMomList,
						   int padding = 0, int sign = -1, int nTime=1, int tId=0){
    int cacheIndex = threadIdx.x;
    int leftToReduce = (int)(gridDim.x/(double)nTime);
    int nMoms = texMomList.Nmoms;
    int id[3] = GET_ID_ZYX(sid3D);
    #pragma unroll
    for(int i=0; i<3; i++) {
      id[i] += DGC_procPosition[i] * DGC_localL[i] - sp[i];
    }
    
    Float phase;
    Float2<Float> expon;
    for(int imom = 0 ; imom < nMoms ; imom++){
      int4 momv = texMomList.get(imom);
      phase = momv.x*id[0]/((Float) DGC_totalL[0]) + momv.y*id[1]/((Float) DGC_totalL[1]) + momv.z*id[2]/((Float) DGC_totalL[2]);
      phase *=  2. * PI;
      expon.x = cos(phase);
      expon.y = sign*sin(phase);
      for(int ip = 0 ; ip < n_comp ; ip++){
      	shared_cache[ip*blockDim.x + cacheIndex] = in[ip] * expon; 
   //     if (nTime==5 || nTime==2 || nTime==1){
   //       printf("InsideBL ip %d blockDim.x %d cacheIndex %d %e %e\n", ip, blockDim.x, cacheIndex, shared_cache[ip*blockDim.x + cacheIndex].givex(),shared_cache[ip*blockDim.x + cacheIndex].givey());
   //  	}
      }
      reduce(shared_cache,n_comp);
      
      if(cacheIndex == 0 && out!=NULL){
	for(int ip = 0 ; ip < n_comp ; ip++){
	  out[((tId*nMoms + imom)*(n_comp+padding) + ip)*leftToReduce + (blockIdx.x%leftToReduce)] =
	    shared_cache[ip*blockDim.x];
//	  if (nTime==5 || nTime==2 || nTime==1){
//	    printf("InsideFT %d gridDim/Timestep %e Blockdim %e tId %d nMoms %d imom %d n_comp %d paddding %d ip %d %e %e\n",leftToReduce, gridDim.x/(double)nTime, (double)blockDim.x, tId, nMoms, imom, n_comp, padding, ip, shared_cache[ip*blockDim.x].givex(),shared_cache[ip*blockDim.x].givey());
//	  }
	}
      }
    }    
  }

  template<typename Float>
  __inline__ __device__ Float xi0(Float w)
  {
    Float tmp,tmp1;
    if(fabs(w)<0.05)
      {
	tmp=1.0-w*w/42.0;
	tmp1=1.0-w*w/20.0*tmp;
	tmp=1.0-w*w/6.0*tmp1;
	return tmp;
      }
    else
      return sin(w)/w;
  }
  __inline__ int get_time_step(int grid, int block){
      int nblocks= (HGC_localVolume3D+block-1)/block;
      return  grid/nblocks;
  }

  template<typename Float>
  __inline__ __device__ void exponentiate_iQ(Float2<Float> a[N_COLS][N_COLS]){

    Float2<Float> M2[N_COLS][N_COLS],M3[N_COLS][N_COLS],Id[N_COLS][N_COLS];
    Float t1,t2,c0max,theta,u,w, factor;
    Float2<Float> C,C1,tmp,tmp1,h0,h1,h2,f0,f1,f2;
    int sign=0;
    t1=real_trace_mul_G_G(a,a);
    mul_G_G(M2,a,a);
    mul_G_G(M3,M2,a);
    t2=real_trace_mul_G_G(M2,a);
    t1/=(Float)2.;
    t2/=(Float)3.;

    if(t2<0)
      {
	sign=1;
	t2=-t2;
      }
    c0max=2.0*pow(t1/((Float)3.0),1.5);
    theta=acos(t2/c0max);
    u=sqrt(t1/((Float)3.0))*cos(theta/((Float)3.0));
    w=sqrt(t1)*sin(theta/((Float)3.0));

    C.x=cos(2.0*u);   C.y=sin(2.0*u);
    C1.x=cos(u);   C1.y=-sin(u);

    tmp.x=u*u-w*w; tmp.y=0.0;
    tmp1.x=8*u*u*cos(w); tmp1.y=2*u*(3.0*u*u+w*w)*xi0(w);

    h0=tmp*C+C1*tmp1;

    tmp.x=2.0*u; tmp.y=0;
    tmp1.x=2.0*u*cos(w); tmp1.y=-(3*u*u-w*w)*xi0(w);
    h1=tmp*C-C1*tmp1;

    tmp.x=cos(w); tmp.y=3.0*u*xi0(w);
    h2=C-C1*tmp;

    factor=9.0*u*u-w*w;
    if(sign==0)
      {
	f0=h0*(1/factor);
	f1=h1*(1/factor);
	f2=h2*(1/factor);
      }
    else
      {
	f0=conj(h0*(1/factor));
	f1=(-1.0)*conj(h1*(1/factor));
	f2=conj(h2*(1/factor));
      }

#pragma unroll
    for(int i=0;i<N_COLS;i++)
#pragma unroll
      for(int j=0;j<N_COLS;j++)
	{
	  if(i==j) Id[i][j].x=1;
	  else Id[i][j].x=0;
	  Id[i][j].y=0;
	}

#pragma unroll
    for(int i=0;i<N_COLS;i++)
#pragma unroll
      for(int j=0;j<N_COLS;j++)
	{
	  a[i][j]=f0*Id[i][j]+f1*a[i][j]+f2*M2[i][j];
	}
  }

  template<typename Float>
  __inline__ __device__ void eigvalsHermTraceless(Float e[N_COLS],Float2<Float> H[N_COLS][N_COLS]){
    // computes eigenvalues of Hermitian traceless matrix
    Float2<Float> ThirdRootOne = {1.,sqrt(3.)};
    Float ThirdRoot_12 = pow(12.,1./3.);
    Float ThirdRoot_18 = pow(18.,1./3.);
    Float ThirdRoot_2_3 = pow((2./3.),1./3.);

    Float a = - ( norm2(H[0][1]) + norm2(H[0][2]) + norm2(H[1][2]) + H[2][2].x * H[2][2].x - H[0][0].x * H[1][1].x ); 
    Float2<Float> b;
    b.x = - H[0][0].x * H[1][1].x * H[2][2].x + H[2][2].x * norm2(H[0][1])
      - (H[0][1] * H[1][2] * conj(H[0][2])).x + H[1][1].x * norm2(H[0][2]);
    b.y = H[2][2].x *(H[0][1] * conj(H[0][1])).y
      - (H[0][1] * H[1][2] * conj(H[0][2])).y + H[1][1].x * (H[0][2] * conj(H[0][2])).y;
    b.x +=   H[0][0].x * (H[1][2] * conj(H[1][2])).x - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).x;               
    b.y +=   H[0][0].x * (H[1][2] * conj(H[1][2])).y - (H[0][2] * conj(H[0][1]) * conj(H[1][2]) ).y;

    Float2<Float> temp1 = {12. * a * a * a + 81. * (b * b).x , 81. * (b * b).y};
    Float2<Float> w = cpow<Float>(temp1,0.5);    
    temp1 = -9. * b + w;
    Float2<Float> D = cpow<Float>(temp1,1./3.);

    temp1.x = a*ThirdRoot_2_3; temp1.y = 0.;
    e[0] = D.x / (ThirdRoot_18) - (temp1/D).x;
    temp1.x = D.x * ThirdRoot_12 ; temp1.y = D.y * ThirdRoot_12;
    e[1] = a * (ThirdRootOne / temp1).x - (conj(ThirdRootOne) * D).x / (ThirdRoot_18*2.);
    e[2] = -e[0]-e[1];
  }

  template<typename Float>
  __inline__ __device__ void normalizeUnitary(Float2<Float> v[N_COLS][N_COLS]){
    Float norma;
    norma = norm2(v[0][0]) + norm2(v[0][1]) + norm2(v[0][2]);
    Float2<Float> w = (v[0][0] * conj(v[1][0]) + v[0][1] * conj(v[1][1]) + v[0][2] * conj(v[1][2]))/norma;

#pragma unroll
    for(int c1 = 0; c1 < N_COLS ; c1++)
      v[1][c1] = v[1][c1] - w * v[0][c1];
    
    norma=1./sqrt(norma);
#pragma unroll
    for(int c1 = 0; c1 < N_COLS ; c1++)
      v[0][c1] = norma*v[0][c1];

    norma = 1./sqrt(norm2(v[1][0]) + norm2(v[1][1]) + norm2(v[1][2]));
#pragma unroll
    for(int c1 = 0; c1 < N_COLS ; c1++)
      v[1][c1] = norma*v[1][c1];

    /////////////////////////////
#pragma unroll
    for(int c1 = 0; c1 < N_COLS ; c1++){
      int r1 = (c1+1)%3;
      int r2 = (c1+2)%3;
      v[2][c1] =conj(v[0][r1]*v[1][r2]) - conj(v[0][r2]*v[1][r1]);
    }
  }
  
  template<typename Float>
  __inline__ __device__ void eigvecsHermTraceless(Float2<Float> v[N_COLS][N_COLS],Float e[N_COLS],Float2<Float> H[N_COLS][N_COLS]){
    // computes eigenvectors of Hermitian traceless matrix having eigenvalues
    v[0][0].x = -(e[0]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);                         
    v[0][0].y = -(e[0]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[0][1].x = -((H[2][0]*H[0][1]).x + e[0]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[0][1].y = -((H[2][0]*H[0][1]).y + e[0]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[0][2].x =-e[0]*e[0] + e[0]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[0]*H[1][1].x - H[0][0].x*H[1][1].x;
    v[0][2].y = 0.;

    v[1][0].x = -(e[1]*H[2][0].x - H[2][0].x*H[1][1].x + (H[1][0]*H[2][1]).x);
    v[1][0].y = -(e[1]*H[2][0].y - H[2][0].y*H[1][1].x + (H[1][0]*H[2][1]).y);

    v[1][1].x = -((H[2][0]*H[0][1]).x + e[1]*H[2][1].x - H[0][0].x*H[2][1].x);
    v[1][1].y = -((H[2][0]*H[0][1]).y + e[1]*H[2][1].y - H[0][0].x*H[2][1].y);

    v[1][2].x =-e[1]*e[1] + e[1]*H[0][0].x + (H[0][1]*conj(H[0][1])).x + e[1]*H[1][1].x - H[0][0].x*H[1][1].x;;
    v[1][2].y = 0.;
    normalizeUnitary(v);
  }
  
  __inline__ __device__ int LEXIC_1DL_1DG(int sid) {
    //compute 4 space-time GLOBID  
    
    int skipvol = 1;
    int globid = 0;
    int TZYX_local[N_DIMS];
    int TZYX_global[N_DIMS];
    
    //creating array from fastest to slowest
    for(int i = 0; i < N_DIMS; ++i){
      TZYX_local[i] = (sid/skipvol) % DGC_localL[i];
      skipvol *= DGC_localL[i];
      //TZYX_global[i] = TZYX_local[i];
    }
    for(int i = 0; i < N_DIMS; ++i)
      TZYX_global[i] = TZYX_local[i] + DGC_procPosition[i] * DGC_localL[i];
    
    for(int i = N_DIMS-1; i>=0; i--)
      globid = globid * DGC_totalL[i] + TZYX_global[i];

    return globid;
  }

  template<typename Float>
  __inline__ __device__ void zero_G(Float2<Float> G[N_COLS][N_COLS]){
#pragma unroll
    for(int c1=0; c1<N_COLS; c1++)
#pragma unroll
      for(int c2=0; c2<N_COLS; c2++){
	G[c1][c2]=0.;
      }
  }
  
}

//---------------------------//
// Used in Plegma_topocharge |
//---------------------------\\

template<typename Float>
__inline__ __device__ void init_to_zero(Float2<Float> a[N_COLS][N_COLS]){
  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      a[i][j].x = (Float) 0.;
      a[i][j].y = (Float) 0.;
    }
  }
}

template<typename FloatA, typename FloatB>
__inline__ __device__ FloatA trace_mul_ImG_ImG(Float2<FloatA> a[N_COLS][N_COLS], Float2<FloatB> b[N_COLS][N_COLS]){
  FloatA tr=0.;

  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++) {
      tr+=( ( a[i][j]-conj(a[j][i]) )*( b[j][i]-conj(b[i][j]) ) ).x;
    }
  }
  return -tr/4.;
}

template<typename FloatA>
__inline__ __device__ void A_copyB( Float2<FloatA> A[N_COLS][N_COLS], Float2<FloatA> B[N_COLS][N_COLS] ){
  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++){
      A[i][j] = B[i][j];
    }
  }
}

template<typename FloatA>
__inline__ __device__ void AntiHermTrless_G(Float2<FloatA> a[N_COLS][N_COLS]){  
  Float2<FloatA> M[N_COLS][N_COLS];
  Float2<FloatA> aux, trace;
  
  trace.x = 0.0;
  trace.y = 0.0;
  //M=a
  
  A_copyB( M, a);
  
  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++){
      aux = M[i][j]-conj(M[j][i]);
      aux = aux/((FloatA)2.);
      a[i][j] = aux;
    }
    trace += a[i][i];
  }

  trace = trace/((FloatA) N_COLS);

  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    a[i][i] -= trace;
  }
  
}

// Exponentiate a matrix G by Taylor expanding the exponential up to maxdeg=10
template<typename FloatA>
__inline__ __device__ void exp_G_Taylor(Float2<FloatA> a[N_COLS][N_COLS]){  
  const int maxdeg = 10;
  Float2<FloatA> A[N_COLS][N_COLS];
  Float2<FloatA> tmp[N_COLS][N_COLS];
  
  A_copyB(A, a);
  scaleG(A, 1.0/((FloatA)maxdeg));
  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    A[i][i] = A[i][i] + 1.0;
  }
  // now A=1+aux/maxdeg                                                                      

  #pragma unroll
  for(int j=maxdeg-1; j>0; j--){
    mul_G_G(tmp, A, a);
    scaleG(tmp, 1.0/((FloatA)j) );
    A_copyB(A, tmp);
    #pragma unroll
    for(int i=0; i<N_COLS; i++){
      A[i][i] = A[i][i] + 1.0;
    }
  }
  A_copyB(a, A);
}

template<typename FloatG>
__inline__ __device__ int check_unitarity( Float2<FloatG> U[3][3] ){
  Float2<FloatG> aux[3][3];
  int ris=0;

  mul_G_Gdag( aux, U, U );  
  #pragma unroll
  for(int i=0; i<N_COLS; i++){
    #pragma unroll
    for(int j=0; j<N_COLS; j++){
      if(i==j){
	aux[i][j] = aux[i][j] - (double)1.0;
      }
      if( norm( aux[i][j] ) > 1e-10 ){
	ris = 1;
      }
    }
  }
  
  if(ris==0){
    if( det(U).x < -0.5 ){
      ris=1;
    }
  }
  
  return ris;
}

template<typename FloatG>
__inline__ __device__ void unitarize_G( Float2<FloatG> U[3][3]){
  Float2<FloatG> c[N_COLS];
  FloatG norm;
  
  for( int i=0; i<N_COLS; i++ ){                                                                 
    for( int j=0; j<i; j++ ){
      c[j].x = 0.0;
      c[j].y = 0.0;
      for( int k=0; k<N_COLS; k++){
	c[j] += U[i][k]*conj(U[j][k]);
      }
    }

    // orthogonalize with respect to previous lines                                                       
    for( int j=0; j<i; j++ ){
      for( int k=0; k<N_COLS; k++ ){
	U[i][k] -= c[j]*U[j][k];
      }
    }

    // normalize the line                                                                                
    norm = 0.0;
    for( int k=0; k<N_COLS; k++){
        norm += norm2(U[i][k]);
    }
    norm = 1.0/sqrt(norm);
    for(int k=0; k<N_COLS; k++){
      U[i][k] =  U[i][k]*norm;
    }
  }

  norm = det(U).x;
  if( norm <= -0.5 ){
    for(int i=0; i<N_COLS; i++){
      U[N_COLS-1][i] = -1.0*U[N_COLS-1][i];
    }
  }
}

template<typename FloatG>
__inline__ __device__ void enforce_unitarity( Float2<FloatG> U[3][3]){
  int cc=0;
  while( check_unitarity(U)==1&&(cc<10000) ){
    unitarize_G( U );
    cc++;
  }
}


#endif
