#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;

template<typename Float>
static __global__ void contractG5_bilinear_kernel(generic2<Float> qLoops, vectorTex<Float> x_l,
						  vectorTex<Float> x_r, Float accum_sign) {
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= qLoops.volume()) return;

  Float2<Float> vl[N_SPINS][N_COLS];
  Float2<Float> vl_dag_g5[N_SPINS][N_COLS];
  Float2<Float> vr[N_SPINS][N_COLS];
  x_l.get(vl,sid);
  Vdag_g5(vl_dag_g5, vl);
  x_r.get(vr,sid);

  Float2<Float> accum;
  
  qLoops.setSid(sid);
  #pragma unroll
  for(int mu = 0 ; mu < N_SPINS ; mu++)
#pragma	unroll
    for(int nu = 0 ; nu < N_SPINS ; nu++){
      accum = (Float2<Float>) {0.,0.};
#pragma unroll
      for(int c1 = 0 ; c1 < N_COLS ; c1++)
	accum = accum + vl_dag_g5[mu][c1] * vr[nu][c1];
      int ind=(nu + mu*N_SPINS);
      qLoops[ind]+=accum_sign*accum;
    }
}

template<typename Float>
static void contractG5_bilinear( generic2<Float> qLoops, vectorTex<Float>& v_l, vectorTex<Float>& v_r, Float accum_sign){
  ProfileStruct ps(qLoops.volume());
  run(ps, "contractG5_bilinear_kernel", contractG5_bilinear_kernel<Float>, qLoops, v_l, v_r, accum_sign);
  checkCudaError();
}
