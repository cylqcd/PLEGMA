#include <PLEGMA_kernel_utils.cuh>
using namespace plegma;
template<typename FloatOut, typename FloatIn, typename FloatGauge>
__global__ void gaussian_smearing_kernel(FloatOut* out,
					 vectorTex<FloatIn> vecInTex, 
					 gaugeTex<FloatGauge> gaugeTex){
  int sid = blockIdx.x*blockDim.x + threadIdx.x;
  if (sid >= c_threads) return;

  Float2<FloatGauge> G[N_COLS][N_COLS];
  Float2<FloatIn> S[N_SPINS][N_COLS], P1[N_SPINS][N_COLS], P2[N_SPINS][N_COLS];
  Float2<FloatOut> tmp[N_SPINS][N_COLS], *out2 = (Float2<FloatOut>*) out;

  #pragma unroll
  for(int mu=0; mu<N_SPINS; mu++) 
    #pragma unroll
    for(int c=0; c<N_COLS; c++)
      tmp[mu][c] = 0.;

  // we don't smear the time -> N_DIMS-1
  #pragma unroll
  for(int dir = 0; dir < N_DIMS-1; dir++) {
    gaugeTex.get(G, dir, sid);
    vecInTex.get<Plus>(S, sid, dir);

    mul_G_V(P1,G,S);

    gaugeTex.get<Minus>(G, dir, sid, dir);
    vecInTex.get<Minus>(S, sid, dir);
    
    mul_Gdag_V(P2,G,S);

    #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++) 
      #pragma unroll
      for(int c=0; c<N_COLS; c++)
	tmp[mu][c] = tmp[mu][c] + P1[mu][c] + P2[mu][c];
  }

  vecInTex.get(S,sid);

  double normalize;
  normalize = 1./(1. + 6. * c_alphaGauss);

  #pragma unroll
    for(int mu=0; mu<N_SPINS; mu++) 
      for(int c=0; c<N_COLS; c++)
	out2[(mu*N_COLS + c)*c_stride + sid] = normalize * (S[mu][c] + c_alphaGauss * tmp[mu][c]);
}

template<typename FloatOut,typename FloatIn, typename FloatGauge>
static void gaussian_smearing(FloatOut* out,
			      vectorTex<FloatIn> vecInTex, 
			      gaugeTex<FloatGauge> gaugeTex){
  ProfileStruct ps(GK_localVolume);
  tuneAndRun(ps, "gaussian_smearing_kernel", gaussian_smearing_kernel<FloatOut,FloatIn,FloatGauge>, out, vecInTex, gaugeTex);
}
