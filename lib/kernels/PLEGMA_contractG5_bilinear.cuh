#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void contractG5_bilinear_kernel(Float *qLoops, vectorTex<Float> x_l,
						  vectorTex<Float> x_r, Float accum_sign) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<Float> vl[N_SPINS][N_COLS];
  Float2<Float> vl_dag_g5[N_SPINS][N_COLS];
  Float2<Float> vr[N_SPINS][N_COLS];
  x_l.get(vl,sid);
  Vdag_g5(vl_dag_g5, vl);
  x_r.get(vr,sid);

  Float2<Float> accum;
  Float2<Float> *qLoops2 = (Float2<Float>*) qLoops;
  
#pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++)
#pragma	unroll
    for(int nu = 0 ; nu < N_SPINS ; nu++){
      accum = (Float2<Float>) {0.,0.};
#pragma unroll
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	accum = accum + vl_dag_g5[mu][c1] * vr[nu][c1];
      int ind=sid+c_threads*(nu + mu*N_SPINS);
      qLoops2[ind] = qLoops2[ind] + accum_sign*accum;
    }
}

template<typename Float>
static void contractG5_bilinear(Float *qLoops, vectorTex<Float> v_l, vectorTex<Float> v_r, Float accum_sign){
  dim3 blockDim( THREADS_PER_BLOCK , 1, 1);
  dim3 gridDim( (GK_localVolume + blockDim.x -1)/blockDim.x , 1 , 1);
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps,contractG5_bilinear_kernel<Float>,qLoops, v_l, v_r, accum_sign);
  checkCudaError();
}
