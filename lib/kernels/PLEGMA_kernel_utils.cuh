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
#include <PLEGMA_kernel_tuner.cuh>
#include <PLEGMA_gammas.cuh>

#ifndef PLEGMA_KERNEL_UTILS_CUH
#define PLEGMA_KERNEL_UTILS_CUH

#define THREADS_PER_BLOCK 64
//#define TIMING_REPORT

using namespace plegma;

namespace plegma {
  
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
    // for reduction threads per block must be power of 2 ( this is always my case)
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
  __inline__ __device__ void fourier_transform_3D( Float2<Float> *out, Float2<Float> *in, Float2<Float> *shared_cache, int n_comp, int sid3D, int sp[3]){
    int cacheIndex = threadIdx.x;
    int id[3] = GET_ID_ZYX(sid3D);
    #pragma unroll
    for(int i=0; i<3; i++) {
      id[i] += c_procPosition[i] * c_localL[i] - sp[i];
    }
    
    Float phase;
    Float2<Float> expon;
    for(int imom = 0 ; imom < c_Nmoms ; imom++){
      phase = 0.;
      #pragma unroll
      for(int i=0; i<3; i++)
	phase += ((Float) (c_moms[imom][i]*id[i]))/((Float) c_totalL[i]);
      phase *=  2. * PI;
      expon.x = cos(phase);
      expon.y = -sin(phase);
      for(int ip = 0 ; ip < n_comp ; ip++){
	shared_cache[ip*blockDim.x + cacheIndex] = in[ip] * expon; 
      }
      reduce(shared_cache,n_comp);
      
      if(cacheIndex == 0 && out!=NULL){
	for(int ip = 0 ; ip < n_comp ; ip++){
	  out[(imom*n_comp + ip)*gridDim.x + blockIdx.x] = shared_cache[ip*blockDim.x];
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
    t1/=2;
    t2/=3;

    if(t2<0)
      {
	sign=1;
	t2=-t2;
      }
    c0max=2.0*pow(t1/3.0,1.5);
    theta=acos(t2/c0max);
    u=sqrt(t1/3.0)*cos(theta/3.0);
    w=sqrt(t1)*sin(theta/3.0);

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
}
#endif
