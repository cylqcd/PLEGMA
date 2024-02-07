#include "PLEGMA_kernel_utils.cuh"
#include "PLEGMA_kernel_tuner.cuh"
using namespace plegma;

template<typename Float>
__device__ void clover_leaves(Float2<Float> F[N_COLS][N_COLS], gauge2<Float> &u, int mu, int nu, int sid){

  Float2<Float> U1[N_COLS][N_COLS], U2[N_COLS][N_COLS], P[N_COLS][N_COLS];

#pragma unroll
  for(int i=0; i<N_COLS; i++)
#pragma unroll
    for(int j=0; j<N_COLS; j++)
      F[i][j]=0.;

  /**
      --<-- 
     |     |
     v     ^
    x|-->--|
   **/
  // U_\mu(x) * U_\nu(x+\mu) * U^dag_\mu(x+nu) * U^\dag_\nu(x)
  u.get(U1,mu,sid); u.template get<Plus>(U2,nu,sid,mu); mul_G_G(P,U1,U2);
  u.template get<Plus>(U2,mu,sid,nu); mul_G_Gdag(U1,P,U2);
  u.get(U2,nu,sid); mul_G_Gdag(P,U1,U2);
  G_plus_aG( F, P, 1.);

  
    /**
      --<-- 
     |     |
     v     ^
     |-->--|x
   **/
  // U_\nu(x) * U^dag_\mu(x-mu+nu) * U^dag_\nu(x-mu) * U_\mu(x-mu)
  u.get(U1,nu,sid); u.template get<MinusPlus>(U2,mu,sid,mu,nu); mul_G_Gdag(P,U1,U2);
  u.template get<Minus>(U2,nu,sid,mu); mul_G_Gdag(U1,P,U2);
  u.template get<Minus>(U2,mu,sid,mu); mul_G_G(P,U1,U2);
  G_plus_aG( F, P, 1.);


  /**
      --<-- x 
     |     |
     v     ^
     |-->--|
   **/
  //U^\dag_\mu(x-mu) * U^\dag_\nu(x-nu-mu) * U_\mu(x-nu-mu) * U_\nu(x-nu)
  u.template get<Minus>(U1,mu,sid,mu); u.template get<MinusMinus>(U2,nu,sid,nu,mu); mul_Gdag_Gdag(P,U1,U2);
  u.template get<MinusMinus>(U2,mu,sid,nu,mu); mul_G_G(U1,P,U2);
  u.template get<Minus>(U2,nu,sid,nu); mul_G_G(P,U1,U2);
  G_plus_aG( F, P, 1.);

  /**
    x  --<-- 
     |     |
     v     ^
     |-->--|
   **/
  //U^\dag_\nu(x-nu) * U_\mu(x-nu) * U_\nu(x+mu-nu) * U^\dag_\mu(x)
  u.template get<Minus>(U1,nu,sid,nu); u.template get<Minus>(U2,mu,sid,nu); mul_Gdag_G(P,U1,U2);
  u.template get<PlusMinus>(U2,nu,sid,mu,nu); mul_G_G(U1,P,U2);
  u.get(U2,mu,sid); mul_G_Gdag(P,U1,U2);
  G_plus_aG( F, P, 1.);

  Gdag(P,F);
  G_plus_aG(F,P,-1.); // F <- F-F^\dag
}

template<typename Float>
static __global__ void clover_leaves_kernel(pFloat2<Float> fmunu, gauge2<Float> gauge){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> F[N_COLS][N_COLS];
  if (sid >= fmunu.volume()) return;
  int count=0;

  fmunu.setSid(sid);
  for(int mu = 0 ; mu < N_DIMS; mu++){
    for(int nu = mu+1 ; nu < N_DIMS; nu++){
      clover_leaves(F,gauge,mu,nu,sid);
#pragma unroll
      for(int c1 = 0 ; c1 < N_COLS; c1++){
#pragma unroll
	for(int c2 = 0 ; c2 < N_COLS; c2++){
	  fmunu.set((count*N_COLS*N_COLS + c1*N_COLS + c2), F[c1][c2]);
	}
      }
      count++;
    }
  }

}

template<typename Float>
static void clover_leaves_k(PLEGMA_Fmunu<Float> &fmunu, PLEGMA_Gauge<Float> &gauge){
  assert(fmunu.checkVolume(gauge));
  ProfileStruct ps(gauge.Total_length());
  tuneAndRun(ps,"clover_leaves_kernel", clover_leaves_kernel<Float>, toField2<pFloat2>(fmunu), toField2<gauge2>(gauge));
  checkQudaError();
}
