#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;




template<typename Float>
static __global__ void apply_gamma_prop_kernel(short int LF,Float *inOut, short int r){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  prop2<Float> prop(inOut);
  Float2<Float> Sin[N_SPINS][N_SPINS][N_COLS][N_COLS];
  Float2<Float> Sout[N_SPINS][N_SPINS][N_COLS][N_COLS];
  const Float2<float> (*gamma2)[4];
  gamma2=(Float2<float> (*)[4]) plegma::gamma;
  if (sid >= c_threads) return;
  prop.get(Sin,sid);

#pragma unroll
  for(int i=0;i<N_COLS;i++)
#pragma unroll 
    for(int j=0;j<N_COLS;j++)
#pragma unroll
      for(int k=0;k<N_SPINS;k++)
#pragma unroll
      for(int l=0;l<N_SPINS;l++){
      Sout[k][l][i][j].x=0.;
      Sout[k][l][i][j].y=0.;
    }
	
#pragma unroll
  for(int nz = 0; nz < N_SPINS; nz++){
    int mu = (LF == LEFT)? gammaInd[r][nz][0] : gammaInd[r][nz][1];
    int nu = (LF == LEFT)? gammaInd[r][nz][1] : gammaInd[r][nz][0];
#pragma unroll
    for(int c1 = 0; c1 < N_COLS; c1++)
#pragma unroll
      for(int c2 = 0; c2 < N_COLS; c2++)
#pragma unroll
	for(int s =0 ;s < N_SPINS; s++)
	  (LF == 0)? Sout[mu][s][c1][c2] =Sout[mu][s][c1][c2]+ Sin[nu][s][c1][c2]*gamma2[r][nz] : Sout[s][mu][c1][c2] =Sout[s][mu][c1][c2]+ Sin[s][nu][c1][c2]*gamma2[r][nz];
  }

  prop.set(Sout,sid);
}

template<typename Float>
static void apply_gamma_prop(LEFTRIGHT LR, Float *inOut,short int r){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma_prop_kernel<<<gridDim,blockDim>>>(LR,(Float*) inOut, r);
  checkCudaError();
}


template<typename Float>
static __global__ void apply_gamma5_propagator_kernel(Float *inOut){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  Float2<Float> *inOut2 = (Float2<Float> *) inOut;
  if (sid >= c_threads) return;
   
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
	  spinor[(mu+2)%4] = inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*c_stride + sid];
	// replacing
        #pragma unroll
	for(int mu = 0 ; mu < N_SPINS ; mu++)
	  inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*c_stride + sid] = spinor[mu];
  }

}

template<typename Float>
void apply_gamma5_propagator(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_gamma5_propagator_kernel<<<gridDim,blockDim>>>(inOut);
}

template<typename Float>
static __global__ void conjugate_propagator_kernel(Float *inOut){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  #pragma unroll
  for(int i = 0 ; i < N_SPINS*N_SPINS*N_COLS*N_COLS ; i++)
    inOut[(i*c_stride + sid)*2 + 1] *= -1.;
}

template<typename Float>
void conjugate_propagator(Float *inOut){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  conjugate_propagator_kernel<<<gridDim,blockDim>>>(inOut);
  checkCudaError();
}

template<typename Float>
static __global__ void apply_boundaries_kernel(Float *inOut, int t0){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;
  int t = (sid/c_localL[0]/c_localL[1]/c_localL[2]) % c_localL[3];
  t += c_procPosition[3] * c_localL[3];

  if( t < t0 ) {
#pragma unroll
    for(int i = 0 ; i < N_SPINS*N_SPINS*N_COLS*N_COLS ; i++) {
      inOut[(i*c_stride + sid)*2 + 0] *= -1.;
      inOut[(i*c_stride + sid)*2 + 1] *= -1.;
    }
  }
}

template<typename Float>
void apply_boundaries(Float *inOut, int t0){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  apply_boundaries_kernel<<<gridDim,blockDim>>>(inOut,t0);
  checkCudaError();
}


template<typename Float>
static __global__ void rotateToPhysicalBase_kernel(Float *inOut, int sign){

  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

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
	  P[mu][nu] = inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*c_stride + sid];

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
	  inOut2[(((mu*N_SPINS + nu)*N_COLS + c1)*N_COLS + c2)*c_stride + sid] = PT[mu][nu];
    }

}

template<typename Float>
void rotateToPhysicalBase(Float* inOut, int sign){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  rotateToPhysicalBase_kernel<Float><<<gridDim,blockDim>>>((Float*) inOut,sign);
  checkCudaError();
}
