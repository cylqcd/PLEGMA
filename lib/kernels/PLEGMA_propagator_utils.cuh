#include "PLEGMA_kernel_utils.cuh"
using namespace plegma;

template<LEFTRIGHT LF,typename Float>
static __global__ void apply_gamma_prop_kernel(prop2<Float> prop, GAMMAS r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> Sin[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_SPINS][N_COLS][N_COLS];
  if (sid >= prop.volume()) return;
  prop.get(Sin,sid);
  gammaProp<LF>(Sout,Sin,r);
  prop.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_prop(LEFTRIGHT LR, PLEGMA_Propagator<Float>& InOut, GAMMAS r){
  auto prop = toField2<prop2>(InOut);
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (prop.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  switch(LR){
  case(LEFT):
    apply_gamma_prop_kernel<LEFT><<<gridDim,blockDim>>>(prop, r);
    break;
  case(RIGHT):
    apply_gamma_prop_kernel<RIGHT><<<gridDim,blockDim>>>(prop, r);
    break;
  }
  checkQudaError();
}


template<typename Float>
static __global__ void apply_gamma5_propagator_kernel(prop2<Float> prop){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= prop.volume()) return;

  prop.setSid(sid);
  #pragma unroll
  for(int nu = 0 ; nu < N_SPINS ; nu++)
    #pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++)
      #pragma unroll
      for(int c2 = 0 ; c2 < N_COLS ; c2++) {
	Float2<Float> spinor[4];
	// inline shuffling
        #pragma unroll
	for(int mu = 0 ; mu < N_SPINS ; mu++)
	  spinor[(mu+2)%4] = prop.get(mu, nu, c1, c2);
	// replacing
        #pragma unroll
	for(int mu = 0 ; mu < N_SPINS ; mu++)
	  prop.set(mu, nu, c1, c2, spinor[mu]);
      }
}

template<typename Float>
void apply_gamma5_propagator(PLEGMA_Propagator<Float>& inOut){
  auto prop = toField2<prop2>(inOut);
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (prop.volume() + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma5_propagator_kernel<<<gridDim,blockDim>>>(prop);
}

template<typename Float>
static __global__ void apply_boundaries_kernel(Float *inOut, int t0){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC->localVolume) return;
  int t = (sid/DGC->localL[0]/DGC->localL[1]/DGC->localL[2]) % DGC->localL[3];
  t += DGC->procPosition[3] * DGC->localL[3];

  if( t < t0 ) {
#pragma unroll
    for(int i = 0 ; i < N_SPINS*N_SPINS*N_COLS*N_COLS ; i++) {
      inOut[(i*DGC->localVolume + sid)*2 + 0] *= -1.;
      inOut[(i*DGC->localVolume + sid)*2 + 1] *= -1.;
    }
  }
}

template<typename Float>
void apply_boundaries(Float *inOut, int t0){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC.localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_boundaries_kernel<<<gridDim,blockDim>>>(inOut,t0);
  checkQudaError();
}

template<typename Float>
static __global__ void rotateToPhysicalBase_kernel(Float *inOut, int sign){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= DGC->localVolume) return;

  Float2<Float> P[4][4], PT[4][4], sign_imag_unit(sign*1., IMAG),
    *inOut2 = (Float2<Float> *) inOut;

  #pragma unroll
  for(int c1 = 0 ; c1 < N_COLS ; c1++)
    #pragma unroll
    for(int c2 = 0 ; c2 < N_COLS ; c2++){

      // copying
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  P[mu][nu] = inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*DGC->localVolume + sid];

      // shuffling
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  PT[mu][nu] = 0.5 * ( P[mu][nu] - P[(mu+2)%4][(nu+2)%4] + sign_imag_unit * ( P[mu][(nu+2)%4] + P[(mu+2)%4][nu] ));

      // replacing
      #pragma unroll
      for(int mu = 0 ; mu < N_SPINS ; mu++)
        #pragma unroll
	for(int nu = 0 ; nu < N_SPINS; nu++)
	  inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*DGC->localVolume + sid] = PT[mu][nu];
    }

}

template<typename Float>
void rotateToPhysicalBase(Float* inOut, int sign){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (HGC.localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  rotateToPhysicalBase_kernel<Float><<<gridDim,blockDim>>>((Float*) inOut,sign);
  checkQudaError();
}

template<typename Float>
static __global__ void prop_mul_V_Vdag_kernel(prop2<Float> prop, vectorTex<Float> vectex1, vectorTex<Float> vectex2) {

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= prop.volume()) return;

  Float2<Float> vc1[N_SPINS][N_COLS];
  Float2<Float> vc2[N_SPINS][N_COLS];
  Float2<Float> prp[N_SPINS][N_SPINS][N_COLS][N_COLS];

  vectex1.get(vc1,sid);
  vectex2.get(vc2,sid);

#pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++){
#pragma unroll
    for(int nu = 0 ; nu < N_SPINS ; nu++){
#pragma unroll
      for(int a = 0 ; a < N_COLS ; a++){
#pragma unroll
        for(int b = 0 ; b < N_COLS ; b++){
          prp[mu][nu][a][b] = vc1[mu][a] * conj(vc2[nu][b]);
	}
      }
    }
  }

  prop.set(prp,sid);

}


template<typename Float>
static void prop_mul_V_Vdag(prop2<Float> prop, vectorTex<Float>& vectex1, vectorTex<Float>& vectex2){
  ProfileStruct ps(prop.volume());
  run(ps, "prop_mul_V_Vdag_kernel", prop_mul_V_Vdag_kernel<Float>, prop, vectex1, vectex2);
  checkQudaError();
}
      
