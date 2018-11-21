#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename Float>
static __global__ void su3Projection_kernel(Float* S){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  su3_2<Float> RS(S);
  Float2<Float> M[N_COLS][N_COLS] , H[N_COLS][N_COLS] , U[N_COLS][N_COLS], v[N_COLS][N_COLS] , vr[N_COLS][N_COLS];
  Float e[N_COLS];
  // initialize some constants

  RS.get(M,sid);
  Float2<Float> detM = det(M);
  Float phase = atan2(detM.y,detM.x)/3.;
  Float2<Float> expPhase = {cos(phase), sin(phase)};

  mul_Gdag_G(H,M,M); // H^2 = M^dag M
  enforce_herm(H); // enforce hermiticity reducing rounding errors
  
  Float sum = norm(H[0][1])+norm(H[0][2])+norm(H[1][2]); // check for non diagonal elements

  if(sum <= 1e-08){ // if H^2 is already diagonal
#pragma unroll
    for(int c1=0; c1<N_COLS; c1++) e[c1]=1./sqrt(H[c1][c1].x); // trivial eigenvalues

#pragma unroll
    for(int c1=0; c1<N_COLS; c1++)
#pragma unroll
      for(int c2=0; c2<N_COLS; c2++)
	U[c1][c2] = e[c1] * M[c1][c2]; // If H^2 is diagonal eigenvectors are just unity

    RS.set(U,sid);
  }
  else{ // in general H^2 is not diagonal
    //We need to compute H^-1 = v 1/sqrt(e) v^\dag    
    //characteristic equation we have to solve P(\lambda) = \lambda^3 + c \lambda^2 + a \lambda + b = 0
    Float trace = real_trace<Float,Float>(H)/3.;
#pragma unroll
    for(int c1=0; c1<N_COLS; c1++) H[c1][c1].x -= trace; // make it traceless to simplify char. eq. so that c=0
    
    //computation of the eigenvalues
    eigvalsHermTraceless(e,H);

#pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++){ // put back the trace
      e[c1] += trace; H[c1][c1].x += trace;
    }

    // eigenvectors
    eigvecsHermTraceless(v,e,H);

    // put the phase of the determinant in the eigenvalues
#pragma unroll
    for(int c1 = 0 ; c1 < N_COLS ; c1++){
     Float2<Float> b = conj(expPhase) / sqrt(e[c1]);
#pragma unroll
      for(int c2 = 0 ; c2 < N_COLS ; c2++){
	vr[c1][c2] = b*v[c1][c2];
      }
    }

    //compute U = M * v * 1/sqrt(\lambda) * v^dag
    mul_G_Gdag(H,M,v); 
    mul_G_G(U,H,vr);
    // normalize and reconstruct the third column
    normalizeUnitary(U);
    RS.set(U,sid);
  }
  
}

template<typename Float>
static void su3Projection_k(PLEGMA_Su3field<Float> &S){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  su3Projection_kernel<Float><<<gridDim,blockDim>>>(S.D_elem());
  checkCudaError();
}
